#!/usr/bin/env python3
# Author: Philipp Allgeuer
#
# The following developer notes document the design of this node, and assists in understanding and mental correctness checks of the complex multi-threaded code.
#
# Main thread:
#  - Sets everything up, but then spins empty doing nothing forever
#  - When a ROS shutdown has been requested (but not yet completed - preshutdown), the services are explicitly shut down, followed by notifying each thread (in order to break out of any condition variable waits), and then joining the threads
#    The background threads (once out of any waits) check whether a ROS shutdown has been requested, and if so, break out of their main loop
#
# Service threads: handle_speak / handle_speak_cached / speak_items
#  - Atomically appends new request to requests dict, appends to queues, and notifies condition variables, then non-atomically waits for background playback if necessary
#  - Queues receive commands that are always eventually terminated by a stop message (whether due to clear, or persist=False and all played), AFTER which stopped_request=True and no more items are ever enqueued for that speech request again (not even another stop message)
#    Note: This means that stopped_request=True right after the stop messages are enqueued, and always before these messages can possibly be processed in the background threads (as queues are always accessed under lock)
#  - The requests map always contains any new speech request BEFORE a corresponding message is enqueued, and thereby strictly before the message can possibly be received in another thread
#  - Only at most one single cache message will ever be enqueued for a particular request item
#  - Only at most one single play message will ever be enqueued for a particular request item IF persist=False (and if all have been played a clear is triggered, resulting in stop messages)
#  - SUMMARY: Initial service call -> Update requests map -> Send cache/play messages -> Follow-up service calls -> Send cache/play messages -> Send stop messages -> Set stopped_request (atomic with sending stop messages) -> Never touch that speech request again
#  - EFFECT ON next_speech_id: Read/written
#  - EFFECT ON requests map: Constant fields are read, Service fields are read/written, totally new dict key/speech request is added
#  - EFFECT ON queues: Items are appended to the queues
#  - EFFECT ON conditions: Condition variables are notified
#
# Cache thread: run_cache
#  - Loop: Check queue item -> Else check for PENDING speech item -> Cache if have one OR wait until something changes
#  - On the first stop message received for a particular speech request, all items PENDING -> CLEARED, request PENDING -> CACHED/CLEARED depending if all items CLEARED, stopped_cache=True (request is never touched again after that)
#  - In every single loop iteration, if no speech items in a non-stopped PENDING speech request are PENDING, then request PENDING -> CACHED
#  - Select speech item to cache -> Set cached_segments -> Append to cached_segments -> Append last segment to cached_segments and set item PENDING to CACHED -> Send CACHED status message -> Check queue -> Set request PENDING to CACHED if no more PENDING (see line above)
#  - It is never allowed for a new PENDING item to be added (or an existing item to be modified to PENDING) in an existing speech request
#  - SUMMARY: Service thread always creates with PENDING, cache thread is the only place that transitions away from PENDING, CACHED is set as soon as last segment is added to cached_segments
#  - EFFECT ON cache queue: Items are popped out of the queue
#  - EFFECT ON conditions: Cache condition is waited on, play condition is notified
#  - EFFECT ON requests map: Constant fields are read, Cache fields are read/written
#
# Play thread: run_play
#  - Loop: Check queue item -> Wait for and play each segment in turn (in order for the very last segment to play, item must be CACHED as this happens at the same time as appending the last segment to cached_segments)
#  - On the first stop message received for a particular speech request, all items CACHED -> CLEARED, request CACHED -> CLEARED if all items cleared, stopped_play=True (request is never touched again after that)
#  - In every single loop iteration, if any request has the stopped trifecta then it is deleted from the requests map
#  - Select speech item to play -> Wait until cached_segments not None -> Wait for first segment to be cached -> Wait for previous sound to finish -> Play segment -> Wait for next segment to be cached -> ... -> Play last segment -> CACHED to CLEARED if non-persistent -> Wait for last sound to finish -> Set played event
#  - SUMMARY: Play thread is the only place that transitions away from CACHED and/or deletes, items must implicitly be CACHED to finish playing
#  - EFFECT ON play queue: Items are popped out of the queue
#  - EFFECT ON conditions: Play condition is waited on
#  - EFFECT ON events: Request item played events are set
#  - EFFECT ON requests map: Constant fields are read, Play fields are read/written, entire key/speech request is deleted
#
# Condition variables: cache_condition / play_condition
#  - Cache condition is notified whenever ANY of the following has just occurred (under lock):
#    - A new speech request has been added to the requests map
#    - An item or stop message has been added to the cache queue
#    - A ROS shutdown has been requested (now in preshutdown)
#  - Play condition is notified whenever ANY of the following has just occurred (under lock):
#    - A new speech request has been added to the requests map
#    - A speech item has had its cached_segments change and/or cache_status change PENDING -> CACHED
#    - A speech request has had stopped_request or stopped_cache set to true
#    - An item or stop message has been added to the play queue
#    - A ROS shutdown has been requested (now in preshutdown)
#
# Speech status updates (order is guaranteed):
#  - SpeechStatus.CACHED is sent whenever a speech item cache status changed from PENDING to CACHED
#  - SpeechStatus.PLAYING is sent right before the call to play() of the first text segment of a speech item (sent anyway if zero text segments)
#  - SpeechStatus.PLAYED is sent right after the next call to wait() after play() has been called on the last text segment of a speech item (sent anyway if zero text segments)
#  - SpeechStatus.CLEARED is sent whenever a speech item cache status changed from CACHED to CLEARED (all speech items in a request are CLEARED when the request is deleted)

# Imports
from __future__ import annotations
import os
import sys
import time
import queue
import threading
import contextlib
import dataclasses
from enum import Enum, auto
from typing import Optional
import numpy as np
import TTS.utils.manage
import TTS.utils.synthesizer
import sox
import sounddevice
import rospy
import std_msgs.msg
import nicol_demos.msg
import nicol_demos.srv

# Main function
def main():
	rospy.init_node('speech_server')
	server = SpeechServer(
		speech_model=rospy.get_param('~speech_model', 'tts_models/en/vctk/vits'),
		speech_lang=rospy.get_param('~speech_lang', 'en'),
		speech_voice=rospy.get_param('~speech_voice', 'p336'),
		speech_tempo=rospy.get_param('~speech_tempo', 0.92),
		item_pause=rospy.get_param('~item_pause', 1.0),
		segment_pause=rospy.get_param('~segment_pause', 0.4),
		use_cuda=rospy.get_param('~use_cuda', True),
		preload_model=rospy.get_param('~preload_model', True),
		verbose=rospy.get_param('~verbose', False),
		debug=rospy.get_param('~debug', False),
	)
	server.run()

# Cache status enumeration
# noinspection PyArgumentList
class CacheStatus(Enum):
	PENDING = auto()  # Item is waiting to be cached (caching may already be in progress)
	CACHED = auto()   # Item is currently cached
	CLEARED = auto()  # Item is no longer needed and has been cleared from cache in case it was cached at some point

# Request item class
@dataclasses.dataclass(frozen=True)
class RequestItem:
	STOP = -1        # Constant signifying that the speech request should be stopped and no longer processed
	speech_id: int   # Speech request ID
	item_index: int  # Item index in the corresponding speech request (may be STOP)

# Play request item class
@dataclasses.dataclass(frozen=True)
class PlayRequestItem(RequestItem):
	played: threading.Event = dataclasses.field(default_factory=threading.Event)  # Service/Play: Event that can be waited on to ensure that the request item has finished playing

# Speech request class
@dataclasses.dataclass
class SpeechRequest:
	speech_id: int                 # Constant: Speech request ID
	items: tuple[SpeechItem]       # Constant: Tuple of the speech items contained in this request
	persist: bool                  # Constant: Flag whether speech items in this request should remain cached beyond their first time being played
	stopped_request: bool = False  # Service/Play: Flag whether this speech request has stopped accepting new requests (nothing more will be enqueued, not even STOPs)
	stopped_cache: bool = False    # Cache/Play: Flag whether this speech request has stopped caching items (nothing is currently caching and nothing will be cached in future)
	stopped_play: bool = False     # Play: Flag whether this speech request has stopped playing items (nothing is currently playing and nothing will be played in future)
	cache_status: CacheStatus = CacheStatus.PENDING  # Cache/Play: Lower bound on the cache status of any contained speech item (PENDING -> CACHED -> CLEARED)

# Speech item class
@dataclasses.dataclass
class SpeechItem:

	speech_id: int              # Constant: Speech request ID that this item belongs to
	item_index: int             # Constant: Item index in the corresponding speech request
	text: str                   # Constant: Text content of this speech item
	text_segments: tuple[str]   # Constant: Text segments of this speech item
	queued_cache: bool = False  # Service: Flag whether this speech item has at some point already been enqueued for caching
	queued_play: bool = False   # Service: Flag whether this speech item has at some point already been enqueued for playing
	cache_status: CacheStatus = CacheStatus.PENDING     # Cache/Play: Current cache status (PENDING -> CACHED -> CLEARED)
	cached_segments: Optional[list[np.ndarray]] = None  # Cache/Play: RAM-cached audio wav data (valid if cache status is CACHED)

	def __str__(self):
		return f'R{self.speech_id}:I{self.item_index}'

# Decorator for background thread run methods
def background_thread(name):
	def background_thread_named(run_func):
		def run_func_wrapper(self, *args, **kwargs):
			rospy.loginfo(f"Running {self.__class__.__name__} {name} thread")
			# noinspection PyBroadException
			try:
				run_func(self, *args, **kwargs)
			except Exception:
				rospy.signal_shutdown(f"Exception in {name} thread")
				raise
			finally:
				rospy.loginfo(f"Exiting {self.__class__.__name__} {name} thread")
		return run_func_wrapper
	return background_thread_named

# Speech server class
class SpeechServer:

	PUB_SPEECH_STATUS = '~speech_status'
	SRV_SPEAK = '~speak'
	SRV_SPEAK_CACHED = '~speak_cached'

	### Main thread ###

	def __init__(self, speech_model, speech_lang, speech_voice, speech_tempo, item_pause, segment_pause, use_cuda, preload_model, verbose, debug):

		self.speech_model = speech_model
		self.speech_lang = speech_lang
		self.speech_voice = speech_voice
		self.speech_tempo = speech_tempo
		self.item_pause = item_pause
		self.segment_pause = segment_pause
		self.use_cuda = use_cuda
		self.preload_model = preload_model
		self.verbose = verbose
		self.debug = debug

		self.speech_model_manager = None
		self.speech_synthesizer = None
		self.speech_segmenter = None
		self.speech_tfrm = None

		self.pub_speech_status = None
		self.srv_speak = None
		self.srv_speak_cached = None

		self.lock = threading.Lock()
		self.next_speech_id = 1
		self.requests: dict[int, SpeechRequest] = {}

		self.cache_thread = None
		self.cache_queue = queue.Queue['RequestItem']()
		self.cache_condition = threading.Condition(lock=self.lock)

		self.play_thread = None
		self.play_queue = queue.Queue['PlayRequestItem']()
		self.play_condition = threading.Condition(lock=self.lock)

	def run(self):

		rospy.on_shutdown(self.on_shutdown)

		self.speech_model_manager = TTS.utils.manage.ModelManager(progress_bar=True, verbose=self.verbose)
		rospy.loginfo(f"Loading speech model '{self.speech_model}' to {'CUDA' if self.use_cuda else 'CPU'}...")
		assert self.speech_model.startswith('tts_models/')
		try:
			with suppress_stdout(suppress=not self.verbose):
				model_path, config_path, model_item = self.speech_model_manager.download_model(self.speech_model)
				print(f" > Model path: {model_path}")
		except (ValueError, KeyError):
			raise ValueError(f"Invalid speech model specification: {self.speech_model}")
		with suppress_stdout(suppress=not self.verbose):
			self.speech_synthesizer = SpeechSynthesizer(tts_checkpoint=model_path, tts_config_path=config_path, use_cuda=self.use_cuda)
		rospy.loginfo(f"Using language '{self.speech_lang or '<default>'}', voice '{self.speech_voice or '<default>'}', sample rate {self.speech_synthesizer.sample_rate}Hz")
		self.speech_synthesizer.configure(language_name=self.speech_lang, speaker_name=self.speech_voice, trim_silence=True)
		self.speech_segmenter = self.speech_synthesizer.get_segmenter()
		if self.speech_tempo != 1:
			rospy.loginfo(f"Adjusting speech tempo to {self.speech_tempo * 100:.3g}%")
			self.speech_tfrm = sox.Transformer()
			self.speech_tfrm.tempo(self.speech_tempo)
		rospy.loginfo(f"Using item pause of {self.item_pause:.2f}s and segment pause of {self.segment_pause:.2f}s")

		if self.preload_model:
			rospy.loginfo("Preloading speech model...")
			self.speech_synthesizer.synthesize_segment('Hi')

		self.pub_speech_status = rospy.Publisher(self.PUB_SPEECH_STATUS, nicol_demos.msg.SpeechStatusArray, queue_size=100)
		self.srv_speak = rospy.Service(self.SRV_SPEAK, nicol_demos.srv.SpeakText, self.handle_speak)
		self.srv_speak_cached = rospy.Service(self.SRV_SPEAK_CACHED, nicol_demos.srv.SpeakCachedText, self.handle_speak_cached)

		self.cache_thread = threading.Thread(target=self.run_cache)
		self.cache_thread.start()

		self.play_thread = threading.Thread(target=self.run_play)
		self.play_thread.start()

		rospy.loginfo(f"Running {self.__class__.__name__} spin loop")
		rospy.spin()

	def on_shutdown(self):
		if self.srv_speak:
			self.srv_speak.shutdown()
		if self.srv_speak_cached:
			self.srv_speak_cached.shutdown()
		with self.lock:
			self.cache_condition.notify_all()
			self.play_condition.notify_all()
		if self.cache_thread:
			with contextlib.suppress(RuntimeError):
				self.cache_thread.join()
		if self.play_thread:
			with contextlib.suppress(RuntimeError):
				self.play_thread.join()

	### Service threads ###

	def handle_speak(self, request: nicol_demos.srv.SpeakTextRequest):

		num_items = len(request.texts)
		num_speak_items = len(request.speak_items)

		with self.lock:

			speech_id = self.next_speech_id
			self.next_speech_id += 1
			if self.verbose:
				rospy.loginfo(f"Received new {'persistent' if request.persist else 'non-persistent'} speech request{' (clear after)' if request.clear else ''} with {num_items} items, of which {num_speak_items} should be spoken{' and waited for' if request.wait else ''} => Created speech request {speech_id}")

			# noinspection PyTypeChecker
			speech_items = tuple(SpeechItem(
				speech_id=speech_id,
				item_index=item_index,
				text=text,
				text_segments=tuple(text_segment for text_segment in self.speech_segmenter.segment(text) if any(char.isalnum() for char in text_segment)),
			) for item_index, text in enumerate(request.texts))
			speech_request = SpeechRequest(speech_id=speech_id, items=speech_items, persist=request.persist)
			self.requests[speech_id] = speech_request
			if self.debug:
				self.print_requests_map()

			success, last_play_request = self.speak_items(speech_request, request.speak_items, request.clear, always_notify=True)

		if request.wait and last_play_request:
			last_play_request.played.wait()

		if self.verbose:
			rospy.loginfo(f"Finished enqueuing{' and waiting for' if request.wait else ''} {num_speak_items} cache and {num_items} play items from new speech request {speech_id}{' (post-clear)' if request.clear else ' (ongoing)'}")

		return nicol_demos.srv.SpeakTextResponse(speech_id=speech_id, success=success)

	def handle_speak_cached(self, request: nicol_demos.srv.SpeakCachedTextRequest):

		num_speak_items = len(request.speak_items)
		if self.verbose:
			rospy.loginfo(f"Received follow-up to known speech request {request.speech_id}{' (clear after)' if request.clear else ''} to speak{' and wait for' if request.wait else ''} {num_speak_items} items")

		with self.lock:
			if speech_request := self.requests.get(request.speech_id, None):
				success, last_play_request = self.speak_items(speech_request, request.speak_items, request.clear, always_notify=False)
			else:
				rospy.logwarn(f"Cannot speak items from cleared or unknown speech request {request.speech_id}")
				success, last_play_request = False, None

		if request.wait and last_play_request:
			last_play_request.played.wait()

		if self.verbose:
			rospy.loginfo(f"Finished enqueuing{' and waiting for' if request.wait else ''} {num_speak_items} follow-up play items from speech request {request.speech_id}{' (post-clear)' if request.clear else ' (ongoing)'}")

		return nicol_demos.srv.SpeakCachedTextResponse(success=success)

	def speak_items(self, speech_request, speak_items, clear, always_notify=False):

		assert self.lock.locked()
		if speech_request.stopped_request:
			rospy.logwarn(f"Cannot speak items from stopped speech request {speech_request.speech_id}")
			return False, None

		num_items = len(speech_request.items)
		cache_queue_updated = False
		play_queue_updated = False
		last_play_request = None
		success = True

		for item_index in speak_items:
			if item_index < 0 or item_index >= num_items:
				rospy.logwarn(f"Requested item index {item_index} is out of range [0,{num_items - 1}] for speech request {speech_request.speech_id}")
				success = False
			else:
				speech_item = speech_request.items[item_index]
				if not speech_item.queued_cache:
					self.cache_queue.put_nowait(RequestItem(speech_id=speech_request.speech_id, item_index=item_index))
					speech_item.queued_cache = True
					cache_queue_updated = True
				if speech_item.queued_play and not speech_request.persist:
					rospy.logwarn(f"Cannot play item {item_index} more than once for non-persistent speech request {speech_request.speech_id}")
					success = False
				else:
					self.play_queue.put_nowait(last_play_request := PlayRequestItem(speech_id=speech_request.speech_id, item_index=item_index))
					speech_item.queued_play = True
					play_queue_updated = True

		if clear or (not speech_request.persist and all(speech_item.queued_play for speech_item in speech_request.items)):
			self.cache_queue.put_nowait(RequestItem(speech_id=speech_request.speech_id, item_index=RequestItem.STOP))
			self.play_queue.put_nowait(PlayRequestItem(speech_id=speech_request.speech_id, item_index=RequestItem.STOP))
			if self.verbose:
				rospy.loginfo(f"Service thread has completely finished with speech request {speech_request.speech_id}")
			speech_request.stopped_request = True
			cache_queue_updated = True
			play_queue_updated = True

		if cache_queue_updated or always_notify:
			self.cache_condition.notify_all()
		if play_queue_updated or always_notify:
			self.play_condition.notify_all()

		return success, last_play_request

	### Cache thread ###

	@background_thread('cache')
	def run_cache(self):

		while True:

			with self.lock:
				while True:

					if rospy.core.is_shutdown_requested():
						return

					cache_speech_item = None  # noqa: This is important if cache_queue.get_nowait() throws an exception

					with contextlib.suppress(queue.Empty):
						while True:
							request_item = self.cache_queue.get_nowait()
							speech_request = self.requests.get(request_item.speech_id, None)
							if not speech_request:
								if request_item.item_index == RequestItem.STOP:
									rospy.logwarn(f"Cannot cache-stop cleared or unknown speech request {request_item.speech_id}")
								else:
									rospy.logwarn(f"Cannot cache item {request_item.item_index} from cleared or unknown speech request {request_item.speech_id}")
							elif speech_request.stopped_cache:
								if request_item.item_index == RequestItem.STOP:
									rospy.logwarn(f"Cannot cache-stop stopped speech request {speech_request.speech_id}")
								else:
									rospy.logwarn(f"Cannot cache item {request_item.item_index} from stopped speech request {speech_request.speech_id}")
							elif request_item.item_index == RequestItem.STOP:
								num_never_cached = 0
								all_items_cleared = True
								for speech_item in speech_request.items:
									if speech_item.cache_status == CacheStatus.PENDING:
										speech_item.cache_status = CacheStatus.CLEARED
										num_never_cached += 1
									elif speech_item.cache_status != CacheStatus.CLEARED:
										all_items_cleared = False
								if speech_request.cache_status == CacheStatus.PENDING:
									speech_request.cache_status = CacheStatus.CLEARED if all_items_cleared else CacheStatus.CACHED
									if self.verbose:
										rospy.loginfo(f"Speech request {speech_request.speech_id} will not have any more items cached")
								if self.verbose:
									rospy.loginfo(f"Cache thread has completely finished with speech request {speech_request.speech_id}{f' ({num_never_cached} items were never cached)' if num_never_cached > 0 else ''}")
								speech_request.stopped_cache = True
								self.play_condition.notify_all()
							elif request_item.item_index < 0 or request_item.item_index >= len(speech_request.items):
								rospy.logwarn(f"Cannot cache item with invalid item index {request_item.item_index} for speech request {speech_request.speech_id}")
							else:
								speech_item = speech_request.items[request_item.item_index]
								if speech_item.cache_status == CacheStatus.PENDING:
									cache_speech_item = speech_item
									break
								elif speech_item.cache_status != CacheStatus.CACHED:
									rospy.logwarn(f"Cannot cache {speech_item} as its cache status is already {speech_item.cache_status.name}")

					for speech_request in self.requests.values():
						if speech_request.cache_status == CacheStatus.PENDING and not speech_request.stopped_cache:
							for speech_item in speech_request.items:
								if speech_item.cache_status == CacheStatus.PENDING:
									if not cache_speech_item:
										cache_speech_item = speech_item
									break
							else:
								speech_request.cache_status = CacheStatus.CACHED
								if self.verbose:
									rospy.loginfo(f"Speech request {speech_request.speech_id} has no more items waiting to be cached")

					speech_request, speech_item = None, None  # noqa: We try to avoid possibly artificially delaying garbage collection of speech requests and items that are removed from the requests map
					if cache_speech_item:
						assert cache_speech_item.cache_status == CacheStatus.PENDING
						break

					if self.verbose:
						rospy.loginfo("Cache thread is idle and waiting for items to cache...")
					if self.debug:
						self.print_requests_map()
					self.cache_condition.wait()

			start_time = time.perf_counter()

			num_text_segments = len(cache_speech_item.text_segments)
			if self.verbose:
				rospy.loginfo(f"Caching {cache_speech_item} in {num_text_segments} segments...")

			audio_sample_count = 0
			if cache_speech_item.text_segments:
				last_segment_num = num_text_segments - 1
				for segment_num, text_segment in enumerate(cache_speech_item.text_segments):
					waveform = self.speech_synthesizer.synthesize_segment(text_segment)
					if self.speech_tfrm:
						waveform = self.speech_tfrm.build_array(input_array=waveform, sample_rate_in=self.speech_synthesizer.sample_rate)
					audio_sample_count += waveform.size
					with self.lock:
						assert cache_speech_item.cache_status == CacheStatus.PENDING
						if self.verbose:
							rospy.loginfo(f"Cached {cache_speech_item} segment {segment_num + 1}")
						if segment_num == 0:
							cache_speech_item.cached_segments = [waveform]
						else:
							cache_speech_item.cached_segments.append(waveform)
						if segment_num == last_segment_num:
							cache_speech_item.cache_status = CacheStatus.CACHED
						statuses = [nicol_demos.msg.SpeechStatus(speech_id=cache_speech_item.speech_id, event=nicol_demos.msg.SpeechStatus.CACHED, items=[cache_speech_item.item_index])]
						self.pub_speech_status.publish(nicol_demos.msg.SpeechStatusArray(header=std_msgs.msg.Header(stamp=rospy.Time.now()), statuses=statuses))
						self.play_condition.notify_all()
			else:
				with self.lock:
					assert cache_speech_item.cache_status == CacheStatus.PENDING
					cache_speech_item.cached_segments = []
					cache_speech_item.cache_status = CacheStatus.CACHED
					statuses = [nicol_demos.msg.SpeechStatus(speech_id=cache_speech_item.speech_id, event=nicol_demos.msg.SpeechStatus.CACHED, items=[cache_speech_item.item_index])]
					self.pub_speech_status.publish(nicol_demos.msg.SpeechStatusArray(header=std_msgs.msg.Header(stamp=rospy.Time.now()), statuses=statuses))
					self.play_condition.notify_all()

			if self.verbose:
				rospy.loginfo(f"Finished caching {cache_speech_item} in {time.perf_counter() - start_time:.1f}s for {audio_sample_count / self.speech_synthesizer.sample_rate:.1f}s of audio")

	### Play thread ###

	@background_thread('play')
	def run_play(self):

		segment_pause_waveform = np.zeros(max(round(self.segment_pause * self.speech_synthesizer.sample_rate), 0), dtype=np.float32)
		item_pause_waveform = np.zeros(max(round(self.item_pause * self.speech_synthesizer.sample_rate), 0), dtype=np.float32)

		while True:

			with self.lock:
				while True:

					if rospy.core.is_shutdown_requested():
						return

					play_speech_request = None  # noqa: These are important if play_queue.get_nowait() throws an exception
					play_speech_item = None     # noqa
					play_event = None           # noqa

					with contextlib.suppress(queue.Empty):
						while True:
							request_item = self.play_queue.get_nowait()
							speech_request = self.requests.get(request_item.speech_id, None)
							if not speech_request:
								if request_item.item_index == RequestItem.STOP:
									rospy.logwarn(f"Cannot play-stop cleared or unknown speech request {request_item.speech_id}")
								else:
									rospy.logwarn(f"Cannot play item {request_item.item_index} from cleared or unknown speech request {request_item.speech_id}")
							elif speech_request.stopped_play:
								if request_item.item_index == RequestItem.STOP:
									rospy.logwarn(f"Cannot play-stop stopped speech request {speech_request.speech_id}")
								else:
									rospy.logwarn(f"Cannot play item {request_item.item_index} from stopped speech request {speech_request.speech_id}")
							elif request_item.item_index == RequestItem.STOP:
								items = []
								all_items_cleared = True
								for speech_item in speech_request.items:
									if speech_item.cache_status == CacheStatus.CACHED:
										speech_item.cache_status = CacheStatus.CLEARED
										speech_item.cached_segments = None
										items.append(speech_item.item_index)
									elif speech_item.cache_status != CacheStatus.CLEARED:
										all_items_cleared = False
								if items:
									statuses = [nicol_demos.msg.SpeechStatus(speech_id=speech_request.speech_id, event=nicol_demos.msg.SpeechStatus.CLEARED, items=items)]
									self.pub_speech_status.publish(nicol_demos.msg.SpeechStatusArray(header=std_msgs.msg.Header(stamp=rospy.Time.now()), statuses=statuses))
								if speech_request.cache_status == CacheStatus.CACHED and all_items_cleared:
									speech_request.cache_status = CacheStatus.CLEARED
									if self.verbose:
										rospy.loginfo(f"Speech request {speech_request.speech_id} does not have any more items in cache")
								if self.verbose:
									rospy.loginfo(f"Play thread has completely finished with speech request {speech_request.speech_id}")
								speech_request.stopped_play = True
							elif request_item.item_index < 0 or request_item.item_index >= len(speech_request.items):
								rospy.logwarn(f"Cannot play item with invalid item index {request_item.item_index} for speech request {speech_request.speech_id}")
							else:
								speech_item = speech_request.items[request_item.item_index]
								if speech_item.cache_status == CacheStatus.CLEARED:
									rospy.logwarn(f"Cannot play {speech_item} as its cache status is already {speech_item.cache_status.name}")
								else:
									play_speech_request = speech_request
									play_speech_item = speech_item
									play_event = request_item.played
									break
							request_item.played.set()

					cleared_request = False
					for speech_id, speech_request in tuple(self.requests.items()):
						if speech_request.stopped_request and speech_request.stopped_cache and speech_request.stopped_play:
							assert speech_request.cache_status == CacheStatus.CLEARED and all(speech_item.cache_status == CacheStatus.CLEARED for speech_item in speech_request.items)
							if self.verbose:
								rospy.loginfo(f"Speech request {speech_id} has completely stopped and is being deleted")
							del self.requests[speech_id]
							cleared_request = True
					if self.debug and cleared_request:
						self.print_requests_map()

					speech_request, speech_item = None, None  # noqa: We try to avoid possibly artificially delaying garbage collection of speech requests and items that are removed from the requests map
					if play_speech_item:
						break

					if self.verbose:
						rospy.loginfo("Play thread is idle and waiting for items to play...")
					if self.debug:
						self.print_requests_map()
					self.play_condition.wait()

			num_text_segments = len(play_speech_item.text_segments)
			if self.verbose:
				rospy.loginfo(f"Playing each of the {num_text_segments} segments in {play_speech_item} as soon as they are cached...")

			published_playing = False
			if num_text_segments > 0:

				last_segment_num = num_text_segments - 1
				for segment_num in range(num_text_segments):

					with self.lock:
						while True:

							if rospy.core.is_shutdown_requested():
								return

							if play_speech_item.cached_segments is not None and len(play_speech_item.cached_segments) > segment_num:
								segment_waveform = play_speech_item.cached_segments[segment_num]
								segment_waveform = np.concatenate((segment_waveform, item_pause_waveform if segment_num == last_segment_num else segment_pause_waveform), axis=0, dtype=segment_waveform.dtype)
								break

							if self.verbose:
								rospy.loginfo(f"Play thread is idle and waiting for {play_speech_item} segment {segment_num + 1} to be cached...")
							if self.debug:
								self.print_requests_map()
							self.play_condition.wait()

					sounddevice.wait()
					if segment_num == 0:
						statuses = [nicol_demos.msg.SpeechStatus(speech_id=play_speech_item.speech_id, event=nicol_demos.msg.SpeechStatus.PLAYING, items=[play_speech_item.item_index])]
						self.pub_speech_status.publish(nicol_demos.msg.SpeechStatusArray(header=std_msgs.msg.Header(stamp=rospy.Time.now()), statuses=statuses))
						published_playing = True
					if self.verbose:
						rospy.loginfo(f"Playing {play_speech_item} segment {segment_num + 1}...")
					sounddevice.play(segment_waveform, samplerate=self.speech_synthesizer.sample_rate, blocking=False)

			else:

				with self.lock:
					while True:

						if rospy.core.is_shutdown_requested():
							return

						if play_speech_item.cached_segments is not None:
							break

						if self.verbose:
							rospy.loginfo(f"Play thread is idle and waiting for {play_speech_item} to be cached...")
						if self.debug:
							self.print_requests_map()
						self.play_condition.wait()

			events = [nicol_demos.msg.SpeechStatus.PLAYED] if published_playing else [nicol_demos.msg.SpeechStatus.PLAYING, nicol_demos.msg.SpeechStatus.PLAYED]
			if not play_speech_request.persist:
				with self.lock:
					assert play_speech_item.cache_status == CacheStatus.CACHED
					play_speech_item.cache_status = CacheStatus.CLEARED
					play_speech_item.cached_segments = None
				events.append(nicol_demos.msg.SpeechStatus.CLEARED)
			sounddevice.wait()
			play_event.set()
			statuses = [nicol_demos.msg.SpeechStatus(speech_id=play_speech_item.speech_id, event=event, items=[play_speech_item.item_index]) for event in events]
			self.pub_speech_status.publish(nicol_demos.msg.SpeechStatusArray(header=std_msgs.msg.Header(stamp=rospy.Time.now()), statuses=statuses))

			segment_waveform = None  # noqa: We try to avoid possibly artificially delaying garbage collection of speech waveforms
			if self.verbose:
				rospy.loginfo(f"Finished playing{'' if play_speech_request.persist else ' and clearing'} all {num_text_segments} of the {play_speech_item} segments")

	### Any thread ###

	def print_requests_map(self):
		assert self.lock.locked()
		print(f"Requests map at timestamp: {rospy.Time.now().to_sec() % 100:.3f}s")
		if self.requests:
			for speech_id, speech_request in self.requests.items():
				assert speech_id == speech_request.speech_id
				print(f"  Request {speech_id}: {'Persistent' if speech_request.persist else 'Non-persistent'}, Stopped {speech_request.stopped_request}-{speech_request.stopped_cache}-{speech_request.stopped_play}, Status {speech_request.cache_status.name}")
				for item_index, speech_item in enumerate(speech_request.items):
					assert speech_id == speech_item.speech_id and item_index == speech_item.item_index
					print(f"    Item {item_index}: Queued {speech_item.queued_cache}-{speech_item.queued_play}, Status {speech_item.cache_status.name}")
					for segment_num, text_segment in enumerate(speech_item.text_segments):
						cached_segment = 'Size ' + '\u00D7'.join(str(dim) for dim in speech_item.cached_segments[segment_num].shape) if speech_item.cached_segments and len(speech_item.cached_segments) > segment_num else 'None'
						print(f"      Segment {segment_num + 1}: {text_segment} => {cached_segment}")
		else:
			print("  <Empty>")

# Speech synthesizer class
class SpeechSynthesizer(TTS.utils.synthesizer.Synthesizer):

	def __init__(self, tts_checkpoint, tts_config_path, use_cuda=True):
		super().__init__(tts_checkpoint=tts_checkpoint, tts_config_path=tts_config_path, use_cuda=use_cuda)
		self.sample_rate = self.tts_config.audio.sample_rate
		self.configured_language_id = None
		self.configured_speaker_id = None
		self.configured_speaker_embedding = None
		self.configured_trim_silence = True

	def configure(self, language_name=None, speaker_name=None, trim_silence=True):

		self.configured_language_id = None
		if getattr(self.tts_model, 'language_manager', None):
			if len(self.tts_model.language_manager.name_to_id) == 1:
				self.configured_language_id = next(iter(self.tts_model.language_manager.name_to_id.values()))
			elif language_name:
				self.configured_language_id = self.tts_model.language_manager.name_to_id[language_name]
			else:
				raise ValueError("Multi-language models require a language name to be specified")

		self.configured_speaker_id = None
		self.configured_speaker_embedding = None
		if hasattr(self.tts_model.speaker_manager, "name_to_id"):
			if len(self.tts_model.speaker_manager.name_to_id) == 1:
				self.configured_speaker_id = next(iter(self.tts_model.speaker_manager.name_to_id.values()))
			elif speaker_name:
				if self.tts_config.use_d_vector_file:
					self.configured_speaker_embedding = np.array(self.tts_model.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False))[None, :]
				else:
					self.configured_speaker_id = self.tts_model.speaker_manager.name_to_id[speaker_name]
			else:
				raise ValueError("Multi-speaker models require a speaker name to be specified")

		self.configured_trim_silence = trim_silence

	def get_segmenter(self):
		return self._get_segmenter(lang=self.configured_language_id or 'en')

	def synthesize_segment(self, text):
		return TTS.utils.synthesizer.synthesis(
			model=self.tts_model,
			text=text,
			CONFIG=self.tts_config,
			use_cuda=self.use_cuda,
			speaker_id=self.configured_speaker_id,
			use_griffin_lim=True,
			do_trim_silence=self.configured_trim_silence,
			d_vector=self.configured_speaker_embedding,
			language_id=self.configured_language_id,
		)['wav'].squeeze()

# Context manager to optionally temporarily suppress stdout
@contextlib.contextmanager
def suppress_stdout(suppress=True):
	if suppress:
		with open(os.devnull, 'w') as devnull:
			stdout = sys.stdout
			sys.stdout = devnull
			try:
				yield
			finally:
				sys.stdout = stdout
	else:
		yield

# Run main function
if __name__ == "__main__":
	main()
# EOF
