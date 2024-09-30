# Camera calibration model classes for working with ROS
# Author: Philipp Allgeuer

# Imports
import yaml
import fractions
from typing import Union
import numpy as np
import cv2
import image_geometry
import sensor_msgs.msg

# Pinhole camera model
class PinholeCameraModel(image_geometry.PinholeCameraModel):

	def __init__(self, msg: sensor_msgs.msg.CameraInfo = None):
		super().__init__()

		self.rvec = None
		self.tvec = None
		self.max_radius = None
		self.rvec_rotmat = None
		self.ray_origin = None
		self.eye = None

		self.resolution = None
		self.roi_width = None
		self.roi_height = None
		self.roi_resolution = None

		if msg is not None:
			self.fromCameraInfo(msg)

	def fromFile(self, filename):
		return self.fromCameraInfo(load_calibration_file(filename))

	def fromCameraInfo(self, msg: sensor_msgs.msg.CameraInfo):
		# Note: It is mandatory that the camera calibration was carried out using image pixel coordinates NOT image axis coordinates!

		if msg.distortion_model not in ('plumb_bob', 'rational_polynomial'):
			raise ValueError(f"Unhandled distortion model: {msg.distortion_model}")

		super().fromCameraInfo(msg)

		self.rvec = np.zeros_like(self.D, shape=(3, 1))
		self.tvec = np.zeros_like(self.D, shape=(3, 1))
		self.max_radius = np.inf
		if self.D is not None and self.D.shape[0] > 14:  # Note: Calibrated rvec/tvec values can be passed as (1-indexed) distortion parameters 15-17 and 18-20 respectively, maximum radius can be passed as parameter 21
			rvec_partial = self.D[14:17, :]
			tvec_partial = self.D[17:20, :]
			self.rvec[:rvec_partial.shape[0], :] = rvec_partial
			self.tvec[:tvec_partial.shape[0], :] = tvec_partial
			if self.D.shape[0] > 20:
				self.max_radius = self.D[20, 0]
			self.D = self.D[:14, :]

		self.rvec_rotmat = cv2.Rodrigues(self.rvec)[0]
		self.ray_origin = -self.tvec.T @ self.rvec_rotmat
		self.eye = np.eye(3)

		L = image_geometry.cameramodels.mkmat(3, 3, (
			1 / self.binning_x, 0, (1 - self.binning_x - 2 * self.raw_roi.x_offset) / (2 * self.binning_x),
			0, 1 / self.binning_y, (1 - self.binning_y - 2 * self.raw_roi.y_offset) / (2 * self.binning_y),
			0, 0, 1,
		))
		self.K = np.matmul(L, image_geometry.cameramodels.mkmat(3, 3, msg.K))  # The super class implementation calculates K and P assuming axis coordinates,
		self.P = np.matmul(L, image_geometry.cameramodels.mkmat(3, 4, msg.P))  # but camera calibrations must be in pixel coordinates (unfortunately)...

		self.roi_width = self.raw_roi.width / self.binning_x
		self.roi_height = self.raw_roi.height / self.binning_y
		self.roi_resolution = (self.roi_width, self.roi_height)

		return self

	@staticmethod
	def _apply_scale(value: Union[int, float], scale: Union[int, float, fractions.Fraction]) -> Union[int, float]:
		new_value = value * scale
		new_value_int = int(new_value)
		if new_value_int == new_value:
			return new_value_int
		else:
			return float(new_value)

	def scale_resolution(self, scale: Union[int, float, fractions.Fraction]):
		# scale = Scale to change all calibration resolutions by (greater than 1 => Increase in resolution)

		if self.stamp is None:
			raise ValueError("Cannot scale pinhole camera model resolution as no valid calibration has been loaded yet")
		elif scale == 1:
			return self

		self.width = self._apply_scale(self.width, scale)
		self.height = self._apply_scale(self.height, scale)
		self.resolution = (self.width, self.height)

		self.raw_roi.width = self._apply_scale(self.raw_roi.width, scale)
		self.raw_roi.height = self._apply_scale(self.raw_roi.height, scale)
		self.raw_roi.x_offset = self._apply_scale(self.raw_roi.x_offset, scale)
		self.raw_roi.y_offset = self._apply_scale(self.raw_roi.y_offset, scale)
		self.roi_width = self.raw_roi.width / self.binning_x
		self.roi_height = self.raw_roi.height / self.binning_y
		self.roi_resolution = (self.roi_width, self.roi_height)

		hscale = (scale - 1) / 2
		L = image_geometry.cameramodels.mkmat(3, 3, (scale, 0, hscale, 0, scale, hscale, 0, 0, 1))
		self.full_K = np.matmul(L, self.full_K)
		self.full_P = np.matmul(L, self.full_P)
		self.K = np.matmul(L, self.K)
		self.P = np.matmul(L, self.P)

		return self

	def change_full_resolution(self, full_resolution: tuple[int, int]):
		# full_resolution = New full resolution (width, height)

		width, height = full_resolution
		if not width or not height:
			raise ValueError("New full resolution must be valid")
		elif not self.width or not self.height:
			raise ValueError("Old full resolution must be valid")
		elif width * self.height != self.width * height:
			raise ValueError(f"Cannot change aspect ratio of full resolution: Old {self.width}x{self.height} vs New {width}x{height}")

		if isinstance(width, int) and isinstance(self.width, int):
			scale = fractions.Fraction(numerator=width, denominator=self.width)
		elif isinstance(height, int) and isinstance(self.height, int):
			scale = fractions.Fraction(numerator=height, denominator=self.height)
		else:
			scale = width / self.width

		return self.scale_resolution(scale)

	def change_roi_resolution(self, roi_resolution: tuple[int, int]):
		# roi_resolution = New ROI resolution (width, height)

		roi_width, roi_height = roi_resolution
		if not roi_width or not roi_height:
			raise ValueError("New ROI resolution must be valid")
		elif not self.roi_width or not self.roi_height:
			raise ValueError("Old ROI resolution must be valid")
		elif roi_width * self.roi_height != self.roi_width * roi_height:
			raise ValueError(f"Cannot change aspect ratio of ROI resolution: Old {self.roi_width}x{self.roi_height} vs New {roi_width}x{roi_height}")

		if isinstance(roi_width, int) and isinstance(self.roi_width, int):
			scale = fractions.Fraction(numerator=roi_width, denominator=self.roi_width)
		elif isinstance(roi_height, int) and isinstance(self.roi_height, int):
			scale = fractions.Fraction(numerator=roi_height, denominator=self.roi_height)
		else:
			scale = roi_width / self.roi_width

		return self.scale_resolution(scale)

	def project_points(self, points, full_image=False):
		# points = tuple[float, float, float] or np.ndarray with shape {3, 3x1, Nx3, Nx1x3, Nx3x1} containing the 3D point(s) to project from camera TF coordinates to image pixel coordinates
		# full_image = Whether the image pixel coordinates are returned for the full image or ROI (default)
		# Return the Nx1x2 image pixel coordinates corresponding to the given camera TF coordinates (image origin is centre of top-left pixel in image)

		if isinstance(points, tuple):
			points = np.array(points, dtype=float).reshape((1, 3))
		else:
			points = points.squeeze()
			if points.ndim == 1:
				points = np.expand_dims(points, axis=0)
			points = points.copy()
			if points.ndim != 2 or points.shape[1] != 3:
				raise ValueError(f"Unexpected points shape: {points.shape}")

		np.matmul(points, self.rvec_rotmat.T, out=points)
		np.add(points, self.tvec.T, out=points)
		radius = np.linalg.norm(points[:, 0:2], axis=1) / points[:, 2]
		points[np.logical_or(radius < 0, radius > self.max_radius), 0:2] = np.nan

		return cv2.projectPoints(  # The 'world' frame in this call to cv2.projectPoints() is the camera TF frame, and the 'camera' frame is the camera pinhole frame
			points,                # Specifying the rvec/tvec parameters is equivalent to not specifying them and instead passing the points CWR * WP + CtW = apply(rvec, points) + tvec = cv2.Rodrigues(rvec)[0] @ point + tvec (pseudocode)
			rvec=(0, 0, 0),        # Rotation vector with shape {3, 1x3, 3x1} corresponding to the rotation CWR from camera pinhole to camera TF coordinates (orientation of camera TF frame relative to camera pinhole frame)
			tvec=(0, 0, 0),        # Translation vector with shape {3, 1x3, 3x1} CtW (origin of camera TF frame relative to camera pinhole frame)
			cameraMatrix=self.full_K if full_image else self.K,
			distCoeffs=self.D,
		)[0]

	def cast_pixel_rays(self, pixels, full_image=False):
		# pixels = tuple[float, float] or np.ndarray with shape {2, 2x1, Nx2, Nx1x2, Nx2x1} containing the image pixel coordinates to cast 3D rays of relative to the camera TF frame (must use pixel coordinates NOT axis coordinates)
		# full_image = Whether the supplied image pixel coordinates are relative to the full image or ROI (default)
		# Returns the Nx1x3 camera TF frame ray unit vectors V corresponding to the given image pixel coordinates, and the 1x3 origin point C of all rays
		# All points in 3D space given by C + lambda * V relative to the camera TF frame (for lambda > 0) map to the same pixel in image space

		if isinstance(pixels, tuple):
			pixels = np.array(pixels, dtype=float).reshape((1, 2))
		else:
			pixels = pixels.squeeze()
			if pixels.ndim == 1:
				pixels = np.expand_dims(pixels, axis=0)
			pixels = pixels.copy()
			if pixels.ndim != 2 or pixels.shape[1] != 2:
				raise ValueError(f"Unexpected pixels shape: {pixels.shape}")

		width, height = self.resolution if full_image else self.roi_resolution
		nan_mask = pixels[:, 0] < 0
		nan_mask |= pixels[:, 0] > width - 1
		nan_mask |= pixels[:, 1] < 0
		nan_mask |= pixels[:, 1] > height - 1
		pixels[nan_mask, :] = np.nan

		points = cv2.undistortPointsIter(pixels, cameraMatrix=self.full_K if full_image else self.K, distCoeffs=self.D, R=self.eye, P=self.eye, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.3))
		points = cv2.convertPointsToHomogeneous(points)
		vecs = points @ self.rvec_rotmat
		vecs /= np.linalg.norm(vecs, axis=2, keepdims=True)  # Note: No division by zero is possible here
		return vecs, np.array(self.ray_origin)

# Load camera info from YAML file
def load_calibration_file(filename):
	msg = sensor_msgs.msg.CameraInfo()
	with open(filename, 'r') as file:
		data = yaml.load(file, Loader=yaml.CSafeLoader)
	msg.width = data['image_width']
	msg.height = data['image_height']
	msg.distortion_model = data['distortion_model']
	msg.D = data['distortion_coefficients']['data']
	msg.K = data['camera_matrix']['data']
	msg.R = data['rectification_matrix']['data']
	msg.P = data['projection_matrix']['data']
	return msg
# EOF
