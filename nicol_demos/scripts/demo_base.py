# Base class for all main demo classes
# Author: Philipp Allgeuer

# Imports
import signal
import fractions
import functools
import threading
import collections
from typing import Optional, Union, Iterable, Tuple, Any
import numpy as np
import cv2
import rospy
import tf2_ros
import cv_bridge
import message_filters
import dynamic_reconfigure.server
import sensor_msgs.msg
import geometry_msgs.msg
import camera_models

# Demo base class
class DemoBase:

	camera_image_topics: Tuple[Optional[str], ...]
	camera_info_topics: Tuple[Optional[str], ...]
	camera_info_yamls: Tuple[Optional[str], ...]
	camera_resolutions: Tuple[Optional[Tuple[int, int]], ...]
	camera_listeners: Optional[Tuple[Union[rospy.Subscriber, message_filters.TimeSynchronizer, None], ...]]
	tf_buffer: Optional[tf2_ros.Buffer]

	### Main thread ###

	def __init__(
			self,
			name: Optional[str] = None,
			config_type: Optional[Any] = None,
			camera_image_topics: Optional[Iterable[Optional[str]]] = None,
			camera_info_topics: Union[Iterable[Optional[str]], bool] = False,
			camera_info_yamls: Optional[Iterable[Optional[str]]] = None,
			camera_resolutions: Union[Iterable[Optional[Tuple[int, int]]], Optional[Tuple[int, int]]] = None,
			camera_force_copy: bool = False,
			tf_listener: Union[bool, float] = False,
			tf_camera_qsize: int = 15,
	):
		# name = Name to use for the demo (e.g. 'MyRobotDemo', default is class name)
		# config_type = Dynamic reconfigure config type to use (i.e. *Config)
		# camera_image_topics = Iterable of topics to subscribe to for camera images (None/empty = skip)
		# camera_info_topics = Iterable of topics to subscribe to for camera info or None/empty to ignore (all ignored if False, all auto-constructed from image topic if True, length must match camera_image_topics if iterable provided)
		# camera_info_yamls = Iterable of YAML file paths to override received camera infos with or None/empty to use the received ones (length must match camera_image_topics if iterable provided)
		# camera_resolutions = Iterable of target camera resolutions as (width, height) pairs or None to keep received resolution (None = Keep resolution for all, single value = use for all, length must match camera_image_topics if iterable provided)
		# camera_force_copy = Force the camera images to either be a copy or a rescaled version of the raw received image (in any case not the same object)
		# tf_listener = Whether to listen to TFs (if a strictly positive float then use this as the cache time)
		# tf_camera_qsize = Maximum TF camera image queue size per camera (camera images that are awaiting their corresponding TF transform)

		self.name = name if name is not None else self.__class__.__name__
		rospy.loginfo(f"Initialising {self.name}")

		self.config_type = config_type
		if self.config_type is not None:
			rospy.loginfo(f"Using dynamic reconfigure with config {self.config_type.__name__}")
		self.config_server = self.config = None

		self.camera_image_topics = tuple(camera_image_topics) if camera_image_topics is not None else ()
		self.num_cameras = len(self.camera_image_topics)

		if camera_info_topics is False:
			self.camera_info_topics = tuple(None for _ in range(self.num_cameras))
		elif camera_info_topics is True:
			self.camera_info_topics = tuple((f'{rospy.names.resolve_name(image_topic).rsplit("/", maxsplit=1)[0]}/camera_info' if image_topic else None) for image_topic in self.camera_image_topics)
		else:
			self.camera_info_topics = tuple(camera_info_topics)
		if len(self.camera_info_topics) != self.num_cameras:
			raise ValueError(f"Must have equal number of camera image and info topics ({self.num_cameras} vs {len(self.camera_info_topics)})")

		if camera_info_yamls is None:
			self.camera_info_yamls = tuple(None for _ in range(self.num_cameras))
		else:
			self.camera_info_yamls = tuple(camera_info_yamls)
		if len(self.camera_info_yamls) != self.num_cameras:
			raise ValueError(f"Must have equal number of camera image topics and info YAMLs ({self.num_cameras} vs {len(self.camera_info_yamls)})")

		if camera_resolutions is None or (isinstance(camera_resolutions, tuple) and len(camera_resolutions) == 2 and isinstance(camera_resolutions[0], int)):
			self.camera_resolutions = tuple(camera_resolutions for _ in range(self.num_cameras))
		else:
			self.camera_resolutions = tuple(((resolution[0], resolution[1]) if resolution is not None else None) for resolution in camera_resolutions)
		if len(self.camera_resolutions) != self.num_cameras:
			raise ValueError(f"Must have equal number of camera image topics and resolutions ({self.num_cameras} vs {len(self.camera_resolutions)})")

		self.camera_stamp = [rospy.Time() for _ in range(self.num_cameras)]
		self.camera_force_copy = camera_force_copy
		self.camera_cv_bridge = cv_bridge.CvBridge()
		self.camera_listeners = None

		for c, (image_topic, info_topic, info_yaml, resolution) in enumerate(zip(self.camera_image_topics, self.camera_info_topics, self.camera_info_yamls, self.camera_resolutions)):
			if image_topic:
				rospy.loginfo(f"Camera {c} image: {image_topic}")
				if info_yaml:
					rospy.loginfo(f"Camera {c} info:  {info_yaml}")
				elif info_topic:
					rospy.loginfo(f"Camera {c} info:  {info_topic}")
				rospy.loginfo(f"Camera {c} size:  {f'{resolution[0]}x{resolution[1]}' if resolution is not None else 'Dynamic'}")

		self.tf_cache_time = 5.0
		if isinstance(tf_listener, float):
			if tf_listener > 0:
				self.tf_cache_time = tf_listener
				tf_listener = True
			else:
				tf_listener = False

		self.tf_camera_qsize = max(tf_camera_qsize, 1)
		if tf_listener:
			self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration.from_sec(self.tf_cache_time))
			self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
			self.tf_image_queues = tuple(collections.deque(maxlen=self.tf_camera_qsize) for _ in range(self.num_cameras))
			rospy.loginfo(f"Started TF listener with {self.tf_cache_time:.1f}s cache time")
			if not self.tf_buffer.can_transform(target_frame='base_link', source_frame='world', time=rospy.Time.now(), timeout=rospy.Duration(secs=1)):
				rospy.logwarn("Cannot detect suitable TF transforms")
		else:
			self.tf_buffer = self.tf_listener = None

		self.prepared_run = False
		self.running = False
		self._signals = {}

	def prepare_run(self):
		# Call this prior to run() in order to activate listeners and such (gets called as part of run() if it hasn't already been called)

		self.prepared_run = True

		if self.config_type is not None:
			self.config_server = dynamic_reconfigure.server.Server(self.config_type, self.config_callback)

		camera_listeners = []
		for c, (image_topic, info_topic, info_yaml, resolution) in enumerate(zip(self.camera_image_topics, self.camera_info_topics, self.camera_info_yamls, self.camera_resolutions)):
			if not image_topic:
				camera_listeners.append(None)
			else:
				if not info_topic or info_yaml:
					info_data = camera_models.load_calibration_file(info_yaml) if info_yaml else None
					camera_listener = rospy.Subscriber(image_topic, sensor_msgs.msg.Image, functools.partial(self.camera_callback, info_data=info_data, camera_id=c, resolution=resolution), queue_size=1, buff_size=40000000)
				else:
					sub_image = message_filters.Subscriber(image_topic, sensor_msgs.msg.Image, queue_size=1, buff_size=40000000)
					sub_info = message_filters.Subscriber(info_topic, sensor_msgs.msg.CameraInfo)
					camera_listener = message_filters.TimeSynchronizer((sub_image, sub_info), queue_size=3, reset=True)
					camera_listener.registerCallback(functools.partial(self.camera_callback, camera_id=c, resolution=resolution))
				camera_listeners.append(camera_listener)
				rospy.loginfo(f"Subscribed to camera {c}")
		self.camera_listeners = tuple(camera_listeners)

	def run(self, rate_hz=45.0, use_thread=False):
		# Run the main demo loop (optionally in a background thread)

		if not self.prepared_run:
			self.prepare_run()

		if use_thread:
			thread = threading.Thread(target=self.run_, args=(rate_hz,))
			thread.start()
			self._set_signal_handler()
			# noinspection PyBroadException
			try:
				self.main()
			except Exception:
				rospy.signal_shutdown('error in main')
				raise
			rospy.signal_shutdown('end of main')
		else:
			self.run_(rate_hz)

	def main(self):
		# Run the main demo sequence (ONLY executed if main demo loop is executed in a background thread)
		pass

	def _set_signal_handler(self):
		self._signals[signal.SIGINT] = signal.signal(signal.SIGINT, self._signal_handler)
		self._signals[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._signal_handler)

	def _signal_handler(self, sig, stackframe):
		prev_handler = self._signals.get(sig, None)
		if callable(prev_handler):
			prev_handler(sig, stackframe)
		raise KeyboardInterrupt

	### Main/Run thread ###

	def run_(self, rate_hz):
		rospy.loginfo(f"Running {self.name} at {rate_hz:.1f}Hz")
		rate = rospy.Rate(rate_hz, reset=True)
		self.running = True
		while not rospy.is_shutdown():
			self.step()
			try:
				rate.sleep()
			except rospy.ROSInterruptException:
				break

	def step(self):
		# Called once per main loop step in run()
		pass

	### Config callback threads ###

	# noinspection PyUnusedLocal
	def config_callback(self, config, level):
		self.config = config  # Atomic thread-safe simple assignment
		return config

	### Camera callback threads ###

	def reset_camera_history(self, camera_id: int):
		# Called when the timestamps of a camera jump back in time
		pass

	def camera_callback(self, image_data: sensor_msgs.msg.Image, info_data: Optional[sensor_msgs.msg.CameraInfo], camera_id: int, resolution: Optional[Tuple[int, int]]):
		# Called for each camera image that arrives for a particular camera

		stamp = image_data.header.stamp
		if stamp < self.camera_stamp[camera_id]:
			self.reset_camera_history(camera_id)
		self.camera_stamp[camera_id] = stamp
		frame_id = image_data.header.frame_id
		raw_img = self.camera_cv_bridge.imgmsg_to_cv2(image_data, 'bgr8')
		raw_img_size = (raw_img.shape[1], raw_img.shape[0])

		img_size = resolution if resolution is not None else raw_img_size
		raw_ratio = fractions.Fraction(numerator=raw_img_size[0], denominator=img_size[0])
		if raw_ratio != fractions.Fraction(numerator=raw_img_size[1], denominator=img_size[1]):
			raise ValueError(f"Raw image ({raw_img_size[0]}x{raw_img_size[1]}) and resized working image ({img_size[0]}x{img_size[1]}) must have same aspect ratio")
		elif raw_ratio < 1:
			rospy.logwarn_throttle_identical(86400, f"Raw image ({raw_img_size[0]}x{raw_img_size[1]}) is smaller than resized working image ({img_size[0]}x{img_size[1]})")
		if img_size != raw_img_size:
			img = cv2.resize(raw_img, dsize=img_size)
		else:
			img = raw_img.copy() if self.camera_force_copy else raw_img

		if info_data:
			if info_data.header.stamp.is_zero():
				info_data.header.stamp = stamp
			if not info_data.header.frame_id:
				info_data.header.frame_id = frame_id
		camera_model = info_data and camera_models.PinholeCameraModel(msg=info_data).change_roi_resolution(img_size)
		camera_image_data = dict(camera_id=camera_id, stamp=stamp, frame_id=frame_id, img=img, raw_img=raw_img, raw_ratio=raw_ratio, camera_model=camera_model, camera_tfrm=None)
		if not self.tf_buffer:
			self.process_camera_image(**camera_image_data)
		else:
			tf_image_queue = self.tf_image_queues[camera_id]
			if tf_image_queue and stamp < tf_image_queue[-1]['stamp']:
				tf_image_queue.clear()
			tf_image_queue.append(camera_image_data)
			num_to_clear = 0
			for i, cidata in enumerate(tf_image_queue, start=1):
				cidata_stamp = cidata['stamp']
				if (stamp - cidata_stamp).to_sec() <= self.tf_cache_time:
					try:
						# Attempts to retrieve the TF transform from the world frame to the required frame_id at the required timestamp
						# Note that the numeric translation/rotation contained in the returned transform is of the world csys relative to the camera csys, BUT nonetheless tf2_geometry_msgs.do_transform*() converts world coordinates to camera coordinates!
						cidata['camera_tfrm'] = self.tf_buffer.lookup_transform(target_frame=cidata['frame_id'], source_frame='world', time=cidata_stamp)
					except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, rospy.ROSTimeMovedBackwardsException):
						continue
					self.process_camera_image(**cidata)
				num_to_clear = i
			for _ in range(num_to_clear):
				tf_image_queue.popleft()

	def process_camera_image(self, camera_id: int, stamp: rospy.Time, frame_id: str, img: np.ndarray, raw_img: np.ndarray, raw_ratio: fractions.Fraction, camera_model: Optional[camera_models.PinholeCameraModel], camera_tfrm: Optional[geometry_msgs.msg.TransformStamped]):
		# Note: img and raw_img might be the same object (unless self.camera_force_copy)
		# Note: The camera model (if available) is guaranteed to be at the correct ROI resolution for img (= ROI)
		# Note: This method will be called from different threads for different cameras (fixed thread <-> camera mapping)
		# print(f"CAM{camera_id}={frame_id} {stamp.to_sec() % 100:.3f} {'x'.join(str(dim) for dim in raw_img.shape)}->{'x'.join(str(dim) for dim in img.shape)}{' MODEL' if camera_model else ''}{' TFRM' if camera_tfrm else ''}")
		pass
# EOF
