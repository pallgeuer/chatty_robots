#!/usr/bin/env python3
# Author: Philipp Allgeuer
# Automatic speech recognition server

# Imports
import os
import re
import sys
import copy
import math
import time
import queue
import audioop
import readline  # noqa: Just for completeness, for input() theoretically affected by stderr suppression in Microphone class [https://github.com/spatialaudio/python-sounddevice/issues/166]
import itertools
import threading
import contextlib
import collections
import dataclasses
from enum import Enum, auto
from typing import Optional, Union, Deque
import numpy as np
import pyaudio
import sounddevice
import rospy
import torch
import whisper
import whisper.tokenizer
import nicol_demos.msg
from multi_action_server import MultiActionServer, ActionInfo
from speech_asr_client import StopReason

# Constants
BEEP_FREQ_START = 698.46  # F5 note
BEEP_FREQ_STOP = 587.33   # D5 note
BEEP_SAMPLE_RATE = 44100
BEEP_DURATION = 0.100
BEEP_VOLUME = 0.08

# Main function
def main():
	rospy.init_node('speech_asr')
	server = SpeechASR(
		mic_device=rospy.get_param('~mic_device', ''),                     # Sound device to use as microphone (integer index or search for case-insensitive substring, see python -m sounddevice for list of devices)
		always_capture=rospy.get_param('~always_capture', True),           # Whether to permanently have the microphone running in the background, even if no action is currently running
		sound_start=rospy.get_param('~sound_start', True),                 # Whether to play a sound to indicate whenever the microphone transitions to enabled (listening)
		sound_stop=rospy.get_param('~sound_stop', True),                   # Whether to play a sound to indicate whenever the microphone transitions to disabled (not listening)
		asr_type=rospy.get_param('~asr_type', 'whisper'),                  # ASR model type
		asr_model=rospy.get_param('~asr_model', 'small'),                  # ASR model variant
		audio_lang=rospy.get_param('~audio_lang', 'english'),              # Assumed audio language (auto = detect)
		translate=rospy.get_param('~translate', True),                     # Translate transcribed text to English if it is not already in English (if supported by model)
		preload_model=rospy.get_param('~preload_model', True),             # Whether to preload the model for faster first inference
		use_cuda=rospy.get_param('~use_cuda', True),                       # Whether to use CUDA for ASR
		energy_silence=rospy.get_param('~energy_silence', 65.0),           # Assumed audio chunk energy for non-speech chunks
		energy_start=rospy.get_param('~energy_start', 200.0),              # Start of speech energy threshold
		energy_stop=rospy.get_param('~energy_stop', 120.0),                # Stop speech energy threshold
		energy_dynamic=rospy.get_param('~energy_dynamic', True),           # Whether to dynamically adjust the energy thresholds and silence level
		energy_dynamic_ts90=rospy.get_param('~energy_dynamic_ts90', 1.2),  # 90% settling time of the dynamic energy adjustments
		beep_delay=rospy.get_param('~beep_delay', 0.225),                  # Maximum delay from finished triggering a beep to when it is first heard in the microphone
		starting_duration=rospy.get_param('~starting_duration', 0.125),    # Consistent start speech chunks in order to trigger start of speech
		start_duration=rospy.get_param('~start_duration', 0.25),           # Minimum duration of speech
		stop_duration=rospy.get_param('~stop_duration', 1.2),              # Amount of stopping non-speech in order to finish a phrase
		padding_duration=rospy.get_param('~padding_duration', 0.7),        # Amount of padding to include before and after a phrase to ensure it is all there
		asr_details=rospy.get_param('~asr_details', False),                # Display detailed console output for ASR model runs
		verbose=rospy.get_param('~verbose', False),                        # Display status update console output for events that occur in the ASR server
		debug=rospy.get_param('~debug', False),                            # Display detailed debugging console output
	)
	server.run()

# Request event enumeration
# noinspection PyArgumentList
class RequestEvent(Enum):
	NEW_GOAL = auto()
	CANCEL = auto()

# Speech detector state enumeration
# noinspection PyArgumentList
class DetectorState(Enum):
	INIT = auto()
	STOPPED = auto()
	STARTING = auto()
	STARTED = auto()

# Action state enumeration
# noinspection PyArgumentList
class ActionState(Enum):
	INIT = auto()
	WAIT_FOR_START = auto()
	RECORDING = auto()
	STOPPED = auto()

# Request item class
@dataclasses.dataclass(frozen=True)
class RequestItem:
	action: ActionInfo
	event: RequestEvent

# Inference request item class
@dataclasses.dataclass(frozen=True)
class InferRequestItem:
	action: ActionInfo
	started: bool
	stopped: bool
	stop_reason: StopReason
	listened_duration: float
	recorded_duration: float
	audio: Optional[bytes]  # Note: This is an optional CPU-endian bytes buffer of sample_width bytes per sample representing signed integers in the full 8*sample_width-bit signed range
	infer_pending: threading.Event

# Capture state class
@dataclasses.dataclass
class CaptureState:

	action: ActionInfo                            # Always valid
	min_chunks: int                               # Always valid
	max_chunks: int                               # Always valid
	min_period_chunks: int                        # Always valid
	state: ActionState = ActionState.INIT         # Always valid
	chunk_id: int = -1                            # Valid if state != INIT
	start_listen_id: int = -1                     # Valid if state != INIT
	start_listen: float = math.nan                # Valid if state != INIT
	start_timeout_id: int = -1                    # Valid if state == WAIT_FOR_START
	start_record_id: int = -1                     # Valid if state == RECORDING
	start_record: float = math.nan                # Valid if state == RECORDING
	chunks: Optional[list[bytes]] = None          # Valid if state == RECORDING
	stop_detectable: bool = False                 # Valid if state == RECORDING
	stop: float = math.nan                        # Valid if state == STOPPED
	stop_reason: StopReason = StopReason.UNKNOWN  # Valid if state == STOPPED
	next_feedback_id: int = -1                    # Always valid
	infer_pending: threading.Event = dataclasses.field(default_factory=threading.Event)  # Always valid

	def prefix(self, now):
		return self.action.id if self.state == ActionState.INIT else f"{self.action.id}@{now - self.start_listen:.3f}s:"

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

# Speech ASR class
class SpeechASR:

	### Main thread ###

	def __init__(
		self,
		mic_device,
		always_capture,
		sound_start,
		sound_stop,
		asr_type,
		asr_model,
		audio_lang,
		translate,
		preload_model,
		use_cuda,
		energy_silence,
		energy_start,
		energy_stop,
		energy_dynamic,
		energy_dynamic_ts90,
		beep_delay,
		starting_duration,
		start_duration,
		stop_duration,
		padding_duration,
		asr_details,
		verbose,
		debug,
	):

		self.mic_device = mic_device
		self.always_capture = always_capture
		self.sound_start = sound_start
		self.sound_stop = sound_stop

		self.asr_type = asr_type
		self.asr_model = asr_model
		self.audio_lang = audio_lang
		self.translate = translate
		self.preload_model = preload_model
		self.use_cuda = use_cuda

		self.energy_silence = energy_silence
		self.energy_start = energy_start
		self.energy_stop = energy_stop
		self.energy_dynamic = energy_dynamic
		self.energy_dynamic_ts90 = energy_dynamic_ts90
		self.beep_delay = beep_delay
		self.starting_duration = starting_duration
		self.start_duration = start_duration
		self.stop_duration = stop_duration
		self.padding_duration = padding_duration

		self.asr_details = asr_details
		self.verbose = verbose
		self.debug = debug

		self.model: Optional[ASRModel] = None

		self.action_server = None
		self.lock = threading.RLock()

		self.capture_thread = None
		self.capture_queue = queue.Queue['RequestItem']()
		self.capture_condition = threading.Condition(lock=self.lock)

		self.infer_thread = None
		self.infer_queue = queue.Queue['InferRequestItem']()
		self.infer_condition = threading.Condition(lock=self.lock)

	def run(self):

		rospy.on_shutdown(self.on_shutdown)

		asr_type = self.asr_type.lower()
		if asr_type == 'whisper':
			self.model = WhisperASR(variant=self.asr_model, audio_lang=self.audio_lang, translate=self.translate, preload_model=self.preload_model, use_cuda=self.use_cuda, verbose=self.asr_details)
		else:
			raise ValueError(f"Unknown ASR type: {asr_type}")

		self.action_server = MultiActionServer(ns='speech_asr', ActionSpec=nicol_demos.msg.PerformASRAction, goal_cb=self.perform_asr_goal_callback, cancel_cb=self.perform_asr_cancel_callback)
		self.action_server.start()

		self.capture_thread = threading.Thread(target=self.run_capture)
		self.capture_thread.start()

		self.infer_thread = threading.Thread(target=self.run_infer)
		self.infer_thread.start()

		rospy.loginfo(f"Running {self.__class__.__name__} spin loop")
		rospy.spin()

	def on_shutdown(self):
		if self.action_server:
			self.action_server.stop()
		with self.lock:
			self.capture_condition.notify_all()
			self.infer_condition.notify_all()
		if self.capture_thread:
			with contextlib.suppress(RuntimeError):
				self.capture_thread.join()
		if self.infer_thread:
			with contextlib.suppress(RuntimeError):
				self.infer_thread.join()

	### Callback threads ###

	def perform_asr_goal_callback(self, action: ActionInfo):

		goal: nicol_demos.msg.PerformASRGoal = action.goal  # noqa
		goal.start_timeout = max(goal.start_timeout, 0.0)
		goal.min_duration = max(goal.min_duration, 1.0)
		goal.max_duration = max(goal.max_duration, 0.0)
		goal.min_period = max(goal.min_period, 0.5)

		reject_reason = None
		if goal.live_text and goal.max_duration <= 0:
			reject_reason = "For computational/memory safety, live text with unlimited maximum duration is not permitted"
		if reject_reason:
			if self.verbose:
				rospy.loginfo(f"{action.id}: Rejected with reason: {reject_reason}")
			return False, reject_reason

		with self.lock:
			self.capture_queue.put_nowait(RequestItem(action=action, event=RequestEvent.NEW_GOAL))
			self.capture_condition.notify_all()

		accept_reason = f"Accepted {'auto' if goal.detect_start else 'manual'}-start{f' (timeout {goal.start_timeout:.4g}s)' if goal.detect_start and goal.start_timeout > 0 else ''} {'auto' if goal.detect_stop else 'manual'}-stop {'live ' if goal.live_text else ''}ASR request of duration {goal.min_duration:.4g}-{goal.max_duration if goal.max_duration > 0 else math.inf:.4g}s and {goal.min_period:.4g}s feedback"
		if self.verbose:
			rospy.loginfo(f"{action.id}: {accept_reason}")
		return True, accept_reason

	def perform_asr_cancel_callback(self, action: ActionInfo):
		with self.lock:
			self.capture_queue.put_nowait(RequestItem(action=action, event=RequestEvent.CANCEL))
			self.capture_condition.notify_all()
		if self.verbose:
			rospy.loginfo(f"{action.id}: Manual stop request received from client")
		return True

	### Capture thread ###

	@background_thread('capture')
	def run_capture(self):

		actions: dict[ActionInfo, CaptureState] = {}
		actions_deleter = DelayedDeleter(actions)
		last_actions = None
		microphone_active = False

		start_beep = stop_beep = None
		if self.sound_start or self.sound_stop:
			timestamps = 2 * np.pi * np.linspace(start=0, stop=BEEP_DURATION, num=round(BEEP_DURATION * BEEP_SAMPLE_RATE), endpoint=False)
			if self.sound_start:
				start_beep = BEEP_VOLUME * np.sin(BEEP_FREQ_START * timestamps)
			if self.sound_stop:
				stop_beep = BEEP_VOLUME * np.sin(BEEP_FREQ_STOP * timestamps)

		with Microphone(mic_device=self.mic_device, sample_format=self.model.sample_format, sample_rate=self.model.sample_rate, chunk_size=self.model.chunk_size, queued=True, verbose=self.debug) as microphone:
			microphone.set_queue_maxsize(queue_maxsize=max(math.ceil(1.0 / microphone.used_chunk_time), 2))

			if self.always_capture:
				microphone.start_listening()
				if self.verbose:
					rospy.loginfo("Microphone has started listening...")

			speech_detector = SpeechDetector(
				chunk_time=microphone.used_chunk_time,
				sample_width=microphone.sample_width,
				energy_silence=self.energy_silence,
				energy_start=self.energy_start,
				energy_stop=self.energy_stop,
				energy_dynamic=self.energy_dynamic,
				energy_dynamic_ts90=self.energy_dynamic_ts90,
				starting_duration=self.starting_duration,
				start_duration=self.start_duration,
				stop_duration=self.stop_duration,
				padding_duration=self.padding_duration,
				debug=self.debug,
			)

			rospy.loginfo("Capture thread is looping and ready for audio...")
			while True:

				with self.lock:
					while True:

						if rospy.core.is_shutdown_requested():
							return

						now = time.perf_counter()

						with contextlib.suppress(queue.Empty):
							while True:
								request_item = self.capture_queue.get_nowait()
								action = request_item.action
								state = actions.get(action, None)
								if request_item.event == RequestEvent.NEW_GOAL:
									if state is None:
										if self.verbose:
											rospy.loginfo(f"{action.id}: Received new goal event")
										goal: nicol_demos.msg.PerformASRGoal = action.goal  # noqa
										min_chunks = max(math.ceil(goal.min_duration / microphone.used_chunk_time), 1)
										max_chunks = max(math.floor(goal.max_duration / microphone.used_chunk_time), 1) if goal.max_duration > 0 else 0
										min_period_chunks = max(math.ceil(goal.min_period / microphone.used_chunk_time), 1)
										if self.debug:
											rospy.loginfo(f"{action.id}: Goal has min chunks {min_chunks}, max chunks {max_chunks}, period chunks {min_period_chunks}")
										actions[action] = CaptureState(action=action, min_chunks=min_chunks, max_chunks=max_chunks, min_period_chunks=min_period_chunks)
									else:
										rospy.logwarn(f"{state.prefix(now)} Ignoring unexpected duplicate new goal event")
								elif request_item.event == RequestEvent.CANCEL:
									if state is None:
										if self.verbose:
											rospy.loginfo(f"{action.id}: Ignoring cancel event for non-capturing action")
									else:
										if self.verbose:
											rospy.loginfo(f"{state.prefix(now)} Received cancel event")
										self.stop_action(state=state, reason=StopReason.REQUEST, now=now, microphone=microphone)
										actions_deleter.delete_now(action)
								else:
									rospy.logerr(f"{action.id}: Unrecognised request item event (should never happen): {request_item.event}")
									break  # Should never ever happen, but helps PyCharm's code assistance...

						with actions_deleter:
							for action, state in actions.items():
								if action.has_terminated():
									rospy.logwarn(f"{state.prefix(now)} Unexpected termination of running action")
									self.stop_action(state=state, reason=StopReason.UNKNOWN, now=now, microphone=microphone, infer=False)
									actions_deleter.delete(action)

						if self.debug:
							if actions != last_actions:
								for state in actions.values():
									print(f"\033[32m{state.action.id}: {state.state.name}, First listen #{state.start_listen_id}, Timeout {'#' + format(state.start_timeout_id) if state.start_timeout_id >= 0 else 'never'}, First rec #{state.start_record_id} @ {state.start_record - state.start_listen:.3f}s, {len(state.chunks) if isinstance(state.chunks, list) else '-'}/{state.max_chunks if state.max_chunks > 0 else math.inf} chunks\033[0m")
								last_actions = {action: copy.copy(state) for action, state in actions.items()}

						if microphone.is_listening() or actions:  # Note: If microphone_active is True then microphone.is_listening() must also be True
							break

						if self.verbose:
							rospy.loginfo("Capture thread has microphone disabled and is waiting for something to do...")
						start = time.perf_counter()
						self.capture_condition.wait()
						now = time.perf_counter()
						if self.verbose:
							rospy.loginfo(f"Capture thread woke up after {now - start:.3f}s")

				if not microphone_active and actions:

					lockout_duration = 0.0
					if self.sound_start:
						start = time.perf_counter()
						self.play_sound(sound=start_beep, sample_rate=BEEP_SAMPLE_RATE)
						now = time.perf_counter()
						elapsed = now - start
						lockout_duration = elapsed + BEEP_DURATION + self.beep_delay
						if self.verbose:
							rospy.loginfo(f"Started playing {BEEP_DURATION * 1000:.0f}ms microphone pre-start sound (waited {elapsed * 1000:.0f}ms)")

					speech_detector.reset(lockout_duration=lockout_duration)
					if not self.always_capture:
						microphone.start_listening()
						if self.verbose:
							rospy.loginfo("Microphone has started listening...")
					microphone_active = True

				if microphone.is_listening():

					start = time.perf_counter()
					chunk = microphone.read_chunk()
					now = time.perf_counter()
					chunk_wall_time = now - start
					chunk_size = len(chunk) // microphone.sample_width
					chunk_time = chunk_size / microphone.used_sample_rate

					if rospy.core.is_shutdown_requested():
						break

					if len(chunk) != microphone.used_chunk_size * microphone.sample_width:

						rospy.logwarn(f"Microphone captured unexpected sized chunk ({len(chunk)} != {microphone.used_chunk_size * microphone.sample_width} bytes) signifying an error or end of stream => Stopping all current actions")
						if actions:
							for state in actions.values():
								self.stop_action(state=state, reason=StopReason.UNKNOWN, now=now, microphone=microphone)
							actions.clear()

					elif microphone_active:

						chunk_id, chunk_energy, energy_silence, energy_start, energy_stop, starting_id, detector_state, trim_stop_chunks = speech_detector.process_chunk(chunk)
						if self.debug:
							print(f"\033[34mChunk {chunk_id} size {chunk_size} = {1000 * chunk_time:.0f}ms read took {1000 * chunk_wall_time:.0f}ms with energy {chunk_energy:.0f} (nom {energy_silence:.0f}, start {energy_start:.0f}, stop {energy_stop:.0f}) and SID {starting_id}\033[0m")
						if chunk_wall_time >= 1.5 * chunk_time:
							rospy.logwarn(f"Microphone read took significantly longer wall time {1000 * chunk_wall_time:.0f}ms than chunk time {1000 * chunk_time:.0f}ms")

						if actions:
							with actions_deleter:
								for action, state in actions.items():

									assert state.state != DetectorState.STOPPED  # Stopped actions should never hang around because we are done with them
									goal: nicol_demos.msg.PerformASRGoal = state.action.goal  # noqa
									state.chunk_id = chunk_id

									if state.state == ActionState.INIT:
										state.start_listen_id = chunk_id
										state.start_listen = now
										if goal.detect_start:
											state.start_timeout_id = state.start_listen_id + max(round(goal.start_timeout / microphone.used_chunk_time), 1) - 1 if goal.start_timeout > 0 else -1  # Note: If this chunk ID finishes processing and still in WAIT_FOR_START then timeout
											state.state = ActionState.WAIT_FOR_START
										else:
											state.start_record_id = chunk_id
											state.start_record = now
											state.chunks = []
											state.state = ActionState.RECORDING
										if self.verbose:
											rospy.loginfo(f"{state.prefix(now)} Initialised action to {state.state.name} state at start chunk {state.start_listen_id}")

									if state.state == ActionState.WAIT_FOR_START:
										if (previous_chunks := speech_detector.can_start(first_id=state.start_listen_id)) is not None:
											state.start_record_id = chunk_id - len(previous_chunks)
											state.start_record = now
											state.chunks = previous_chunks
											state.state = ActionState.RECORDING
											if self.verbose:
												rospy.loginfo(f"{state.prefix(now)} Started RECORDING with {len(state.chunks)} initial buffered frames")
										elif starting_id >= state.start_timeout_id >= 0:
											if self.verbose:
												rospy.loginfo(f"{state.prefix(now)} Timed out while waiting for start of speech")
											self.stop_action(state=state, reason=StopReason.TIMEOUT, now=now, microphone=microphone)
											actions_deleter.delete(action)

									if state.state == ActionState.RECORDING:
										state.chunks.append(chunk)
										if detector_state == DetectorState.STARTED and not state.stop_detectable:
											if self.verbose and goal.detect_stop:
												rospy.loginfo(f"{state.prefix(now)} STOP DETECTABLE as detector has seen the STARTED state")
											state.stop_detectable = True
										if goal.detect_stop and state.stop_detectable and detector_state == DetectorState.STOPPED:
											del state.chunks[min(max(len(state.chunks) - trim_stop_chunks, 0), state.max_chunks):]
											if self.verbose:
												rospy.loginfo(f"{state.prefix(now)} DETECTED end of speech with {len(state.chunks)} chunks = {len(state.chunks) * microphone.used_chunk_time:.3f}s")
											self.stop_action(state=state, reason=StopReason.DETECTED, now=now, microphone=microphone)
											actions_deleter.delete(action)
										else:
											if (num_chunks := len(state.chunks)) >= state.max_chunks:
												if num_chunks > state.max_chunks:
													rospy.logwarn(f"{state.prefix(now)} Captured too many chunks ({num_chunks}) for maximum duration chunks ({state.max_chunks}) => Trimming down")
													del state.chunks[state.max_chunks:]
												if self.verbose:
													rospy.loginfo(f"{state.prefix(now)} STOP recording with {len(state.chunks)} chunks = {len(state.chunks) * microphone.used_chunk_time:.3f}s due to maximum duration {goal.max_duration:.4g}s")
												self.stop_action(state=state, reason=StopReason.DURATION, now=now, microphone=microphone)
												actions_deleter.delete(action)

									if state.state != ActionState.STOPPED and chunk_id >= state.next_feedback_id and not state.infer_pending.is_set():
										use_audio = goal.live_text and state.chunks and len(state.chunks) >= state.min_chunks
										if self.verbose:
											rospy.loginfo(f"{state.prefix(now)} Triggering FEEDBACK {f'with {len(state.chunks)} chunks = {len(state.chunks) * microphone.used_chunk_time:.3f}s of' if use_audio else 'without'} audio")
										self.infer_action(state=state, use_audio=use_audio, microphone=microphone)
										state.next_feedback_id = chunk_id + state.min_period_chunks

				if microphone_active and not actions:

					if not self.always_capture:
						microphone.stop_listening()
						if self.verbose:
							rospy.loginfo("Microphone has stopped listening")
					speech_detector.reset()
					microphone_active = False

					if self.sound_stop:
						start = time.perf_counter()
						self.play_sound(sound=stop_beep, sample_rate=BEEP_SAMPLE_RATE)
						now = time.perf_counter()
						if self.verbose:
							rospy.loginfo(f"Started playing {BEEP_DURATION * 1000:.0f}ms microphone post-stop sound (waited {(now - start) * 1000:.0f}ms)")

	def stop_action(self, state, reason, now, microphone, infer=True):
		state.stop = now
		state.stop_reason = reason
		state.state = ActionState.STOPPED
		if self.verbose:
			rospy.loginfo(f"{state.prefix(now)} Action STOPPED with reason {state.stop_reason.name} {f'and {len(state.chunks)} chunks = {len(state.chunks) * microphone.used_chunk_time:.3f}s of' if state.chunks is not None else 'without'} audio")
		if infer:
			self.infer_action(state=state, use_audio=True, microphone=microphone)

	def infer_action(self, state, use_audio, microphone):
		inited = (state.state != ActionState.INIT)
		started = state.chunks is not None
		stopped = (state.state == ActionState.STOPPED)
		listened_duration = (state.chunk_id - state.start_listen_id + 1) * microphone.used_chunk_time if inited else 0.0
		recorded_duration = len(state.chunks) * microphone.used_chunk_time if started else 0.0
		audio = b''.join(state.chunks) if use_audio and started else None
		state.infer_pending.set()
		with self.lock:
			self.infer_queue.put_nowait(InferRequestItem(action=state.action, started=started, stopped=stopped, stop_reason=state.stop_reason, listened_duration=listened_duration, recorded_duration=recorded_duration, audio=audio, infer_pending=state.infer_pending))
			self.infer_condition.notify_all()

	# noinspection PyProtectedMember
	@classmethod
	def play_sound(cls, sound, sample_rate):
		if sounddevice._last_callback:
			status = sounddevice._last_callback.status
			if status.output_underflow:
				rospy.logwarn("Sound playback stream encountered output underflow in the past")
			if status.output_overflow:
				rospy.logwarn("Sound playback stream encountered output overflow in the past")
		sounddevice.wait()
		sounddevice.play(sound, samplerate=sample_rate, blocking=False)

	### Inference thread ###

	@background_thread('inference')
	def run_infer(self):

		pending_infers: Deque[InferRequestItem] = collections.deque()

		while True:

			with self.lock:
				while True:

					if rospy.core.is_shutdown_requested():
						return

					with contextlib.suppress(queue.Empty):
						while True:
							request_item = self.infer_queue.get_nowait()
							if request_item is None:
								rospy.logerr("Received null inference request item (should never happen)")
								break  # Should never ever happen, but helps PyCharm's code assistance...
							if self.verbose:
								rospy.loginfo(f"{request_item.action.id}@{request_item.listened_duration:.3f}s Received {'FINAL' if request_item.stopped else 'FEEDBACK'} inference request {f'with {len(request_item.audio)} bytes = {request_item.recorded_duration:.3f}s audio' if request_item.audio is not None else 'without audio'}")
							if request_item.audio:
								pending_infers.append(request_item)
							else:
								self.respond_action(request_item=request_item, text="" if request_item.audio is not None else None)

					if pending_infers:
						break

					if self.verbose:
						rospy.loginfo("Inference thread is waiting for something to do...")
					start = time.perf_counter()
					self.infer_condition.wait()
					now = time.perf_counter()
					if self.verbose:
						rospy.loginfo(f"Inference thread woke up after {now - start:.3f}s")

			request_item = pending_infers.popleft()
			if self.verbose or self.asr_details:
				rospy.loginfo(f"{request_item.action.id} Performing inference on recorded audio")
			start = time.perf_counter()
			text = self.model(request_item.audio)
			now = time.perf_counter()
			if self.verbose or self.asr_details:
				rospy.loginfo(f"{request_item.action.id} Finished inference on recorded audio in {now - start:.3f}s")
			self.respond_action(request_item=request_item, text=text)

	def respond_action(self, request_item, text):
		if request_item.stopped:
			if request_item.stop_reason in (StopReason.DETECTED, StopReason.DURATION) or (request_item.stop_reason == StopReason.REQUEST and request_item.started):
				func = request_item.action.completed
				response = 'Completed'
			else:
				func = request_item.action.aborted
				response = 'Aborted'
			func(result=nicol_demos.msg.PerformASRResult(
				started=request_item.started,
				stop_reason=request_item.stop_reason.value,
				listened=request_item.listened_duration,
				recorded=request_item.recorded_duration,
				text=text,
			), status=f"{response} ASR request {f'with {len(text)} chars of text' if text is not None else 'without text'}")
		else:
			response = 'Feedback'
			request_item.action.feedback(feedback=nicol_demos.msg.PerformASRFeedback(
				started=request_item.started,
				cur_listened=request_item.listened_duration,
				cur_recorded=request_item.recorded_duration,
				have_text=text is not None,
				cur_text=text,
			))
		request_item.infer_pending.clear()  # Note: This might clear the flag after a feedback message although a stop message has already been enqueued and is still pending, but this doesn't matter because the flag is only used to pace the feedback messages, and the fact that the action is stopped already prevents all further feedback messages
		if self.verbose:
			rospy.loginfo(f"{request_item.action.id} Sent {response.upper()} msg {'with text: ' + (text if len(text) <= 60 else '...' + text[-57:]) if text is not None else 'without text'}")

# ASR model base class
class ASRModel:

	SAMPLE_FORMATS = {
		pyaudio.paInt8: np.int8,
		pyaudio.paInt16: np.int16,
		pyaudio.paInt24: np.int32,
		pyaudio.paInt32: np.int32,
	}

	def __init__(self, sample_format: int, sample_rate: Optional[int], chunk_size: Optional[int]):
		self.sample_format = sample_format
		assert self.sample_format in self.SAMPLE_FORMATS  # Note: Required for audioop
		self.sample_format_np = self.SAMPLE_FORMATS[self.sample_format]
		self.sample_width = pyaudio.get_sample_size(self.sample_format)
		self.sample_scale = 1 << (8 * self.sample_width - 1)  # e.g. 32768 for pyaudio.paInt16
		self.sample_rate = sample_rate
		self.chunk_size = chunk_size

	def __call__(self, audio: bytes) -> str:
		# Note: The audio data is a raw bytes buffer in the specified sample format (e.g. pyaudio.paInt16 = CPU-endian signed 16-bit integers with full -32768 to 32767 range possible)
		raise NotImplementedError

# Whisper ASR class
class WhisperASR(ASRModel):

	def __init__(self, variant='small', audio_lang='english', translate=True, preload_model=True, use_cuda=True, verbose=False):

		audio_lang = audio_lang.lower()
		if audio_lang == 'auto':
			audio_lang = None
		else:
			audio_lang = whisper.tokenizer.LANGUAGES.get(audio_lang, audio_lang)

		# noinspection PyProtectedMember
		variant_url_map = whisper._MODELS
		variant = variant.lower()
		if variant not in variant_url_map:
			raise ValueError(f"Unknown whisper model variant: {variant}")
		if audio_lang == 'english':
			variant_en = variant + '.en'
			if variant_en in variant_url_map:
				variant = variant_en

		device = torch.device('cuda' if use_cuda else 'cpu')
		download_root = os.path.join(os.getenv('XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache')), 'whisper')
		model_path = os.path.join(download_root, os.path.basename(variant_url_map[variant]))
		if torch.cuda.is_available() and not use_cuda:
			rospy.logwarn("Performing model inference on CPU although CUDA is available")
		rospy.loginfo(f"Loading whisper model {variant} to {'CUDA' if use_cuda else 'CPU'}: {model_path}")
		model = whisper.load_model(name=variant, device=device, download_root=download_root, in_memory=True)

		audio_lang_cap = audio_lang.title() if audio_lang is not None else None
		if translate and audio_lang != 'english':
			task = 'translate'
			if audio_lang is None:
				rospy.loginfo("Transcribing and translating multilingual audio to English text")
			else:
				rospy.loginfo(f"Transcribing and translating {audio_lang_cap} audio to English text")
		else:
			task = 'transcribe'
			rospy.loginfo(f"Transcribing {audio_lang_cap} audio to {audio_lang_cap} text")

		self.model = model
		self.model_verbose = True if verbose else None
		self.task = task
		self.audio_lang = audio_lang
		self.use_cuda = use_cuda
		self.detected_lang_codes = {'en'}

		super().__init__(sample_format=pyaudio.paInt16, sample_rate=round(whisper.audio.SAMPLE_RATE), chunk_size=None)

		if preload_model and self.use_cuda:
			rospy.loginfo("Preloading whisper model for faster initial inference time...")
			self(bytes(self.sample_rate * self.sample_width))
			rospy.loginfo("Finished preloading whisper model")

	def __call__(self, audio: bytes) -> str:
		audio_np = np.divide(np.frombuffer(audio, dtype=self.sample_format_np), self.sample_scale, dtype=np.float32)
		result = self.model.transcribe(audio=audio_np, verbose=self.model_verbose, task=self.task, language=self.audio_lang, fp16=self.use_cuda)
		lang_code = result['language']
		if self.audio_lang is None and lang_code not in self.detected_lang_codes:
			rospy.loginfo(f"Detected first use of audio language {whisper.tokenizer.LANGUAGES[lang_code].title()}")
			self.detected_lang_codes.add(lang_code)
		text = ' '.join(result['text'].split())
		if text and re.fullmatch(r"[\s!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]*((thanks for watching|thank you|bye|you)[\s!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]*)?", text, re.IGNORECASE):
			if self.model_verbose:
				print(f"\033[33mSuppressing common bad model output and replacing with empty string: {text}\033[0m")
			text = ""
		return text

# Microphone class
class Microphone:

	### Main thread ###

	def __init__(self, mic_device: Union[str, int, None] = None, sample_format: Optional[int] = None, sample_rate: Optional[int] = None, chunk_size: Optional[int] = None, queued: bool = True, queue_maxsize: int = 16, verbose: bool = False):

		self.mic_device = mic_device
		self.sample_format = sample_format if sample_format is not None else pyaudio.paInt16
		self.sample_width = pyaudio.get_sample_size(self.sample_format)
		self.sample_rate = sample_rate
		self.chunk_size = chunk_size
		self.queued = queued
		self.queue_maxsize = queue_maxsize
		self.verbose = verbose

		self.audio = None
		self.device_index = None
		self.used_sample_rate = None
		self.used_chunk_size = None
		self.used_chunk_time = None

		self.queue = None
		self.stream = None

	def set_queue_maxsize(self, queue_maxsize):
		# Note: This will only affect the next stream if a current stream is already listening
		self.queue_maxsize = queue_maxsize

	def __enter__(self):

		if self.verbose:
			self.audio = pyaudio.PyAudio()
		else:
			with contextlib.ExitStack() as stack:
				devnull = os.open(os.devnull, os.O_WRONLY)
				stack.callback(os.close, devnull)
				old_stderr = os.dup(2)
				stack.callback(os.close, old_stderr)
				sys.stderr.flush()
				os.dup2(devnull, 2)
				stack.callback(os.dup2, old_stderr, 2)
				self.audio = pyaudio.PyAudio()

		# noinspection PyBroadException
		try:

			audio_info = self.audio.get_default_host_api_info()
			audio_device_count = audio_info['deviceCount']
			audio_device_default = audio_info['defaultInputDevice']
			assert 0 <= audio_device_default < audio_device_count
			rospy.loginfo(f"Using default audio host API (host index {audio_info['index']}): {audio_info['name']} => {audio_device_count} sound devices")

			if self.mic_device is None:
				device_index = audio_device_default
			elif isinstance(self.mic_device, int):
				device_index = self.mic_device
			elif isinstance(self.mic_device, str):
				if not self.mic_device:
					device_index = audio_device_default
				else:
					try:
						device_index = int(self.mic_device)
					except ValueError:
						matching_devices = []
						mic_device = self.mic_device.lower()
						for i in range(audio_device_count):
							device_info = self.audio.get_device_info_by_index(i)
							if mic_device in device_info['name'].lower() and device_info['maxInputChannels'] > 0:
								matching_devices.append((i == audio_device_default, i))
						if matching_devices:
							_, device_index = max(matching_devices)
						else:
							rospy.logwarn(f"Failed to find microphone device that matches '{self.mic_device}' => Falling back to default microphone of index {audio_device_default}")
							device_index = audio_device_default
			else:
				raise ValueError(f"Invalid microphone device specification type: {self.mic_device}")

			if device_index < 0 or device_index >= audio_device_count:
				rospy.logwarn(f"Invalid microphone device index {device_index} => Falling back to default microphone of index {audio_device_default}")
				device_index = audio_device_default
			if self.audio.get_device_info_by_index(device_index)['maxInputChannels'] <= 0:
				rospy.logwarn(f"Sound device index {device_index} is not a microphone => Falling back to default microphone of index {audio_device_default}")
				device_index = audio_device_default

			self.device_index = device_index
			device_info = self.audio.get_device_info_by_index(self.device_index)
			rospy.loginfo(f"Using device {self.device_index} as microphone: {device_info['name']}, {audio_info['name']} ({device_info['maxInputChannels']} in, {device_info['maxOutputChannels']} out)")

			self.used_sample_rate = self.sample_rate if self.sample_rate else int(device_info["defaultSampleRate"])
			if self.used_sample_rate <= 0:
				raise ValueError(f"Microphone sample rate must be a positive integer: {self.used_sample_rate}")
			self.used_chunk_size = self.chunk_size if self.chunk_size else 2 ** round(math.log2(0.064 * self.used_sample_rate))
			if self.used_chunk_size <= 0:
				raise ValueError(f"Microphone chunk size must be a positive integer: {self.used_chunk_size}")
			self.used_chunk_time = self.used_chunk_size / self.used_sample_rate

			assert self.sample_format in (pyaudio.paInt8, pyaudio.paInt16, pyaudio.paInt24, pyaudio.paInt32)  # Only for the next line...
			rospy.loginfo(f"Using sample rate {self.used_sample_rate}Hz with {8 * self.sample_width}-bit signed int data and chunk size {self.used_chunk_size} = {1000 * self.used_chunk_time:.0f}ms")

		except Exception:
			self.__exit__(None, None, None)

		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop_listening()
		if self.audio is not None:
			self.audio.terminate()
			self.audio = None
		self.device_index = None
		self.used_sample_rate = None
		self.used_chunk_size = None
		self.used_chunk_time = None

	def is_ready(self):
		return self.audio is not None

	@contextlib.contextmanager
	def listen(self):
		self.start_listening()
		try:
			yield
		finally:
			self.stop_listening()

	def start_listening(self):
		assert self.queue is None and self.stream is None
		if self.queued:
			self.queue = queue.Queue[bytes](maxsize=self.queue_maxsize)
		self.stream: pyaudio.Stream = self.audio.open(
			rate=self.used_sample_rate,
			channels=1,
			format=self.sample_format,
			input=True,
			input_device_index=self.device_index,
			frames_per_buffer=self.used_chunk_size,
			start=True,
			stream_callback=self.stream_callback if self.queued else None,
		)

	def ensure_listening(self):
		if self.stream is None:
			self.start_listening()

	def stop_listening(self):
		if self.stream is None:
			self.queue = None
		else:
			try:
				if not self.stream.is_stopped():
					self.stream.stop_stream()
			except OSError as exc:
				if exc.errno != -9988:  # Note: Suppress [Errno -9988] Stream closed
					raise
			finally:
				try:
					self.stream.close()
				finally:
					self.stream = None
					self.queue = None

	def is_listening(self) -> bool:
		return self.stream is not None

	def read_chunk(self) -> bytes:
		# Note: This method blocks until the next chunk is available (should never take much longer than a single chunk time of wall time)
		if self.queued:
			return self.queue.get()
		else:
			return self.stream.read(num_frames=self.used_chunk_size, exception_on_overflow=False)  # Note: Can temporarily make this True to check that no audio buffers are being overflowed and skipped

	### Audio thread ###

	# noinspection PyUnusedLocal
	def stream_callback(self, in_data, frame_count, time_info, status_flags):
		if status_flags & pyaudio.paInputUnderflow:
			rospy.logwarn("Microphone stream encountered input underflow")
		if status_flags & pyaudio.paInputOverflow:
			rospy.logwarn("Microphone stream encountered input overflow")
		if frame_count <= 0:
			rospy.logwarn("Microphone stream received empty frame")
		try:
			self.queue.put_nowait(in_data)
		except queue.Full:
			try:
				self.queue.get_nowait()  # Note: We want to make sure that a full queue drops the oldest chunks, not the newest one we were about to write
			except queue.Empty:
				pass
			rospy.logwarn(f"Microphone stream encountered full queue (max size {self.queue.maxsize})")
			self.queue.put_nowait(in_data)  # Note: This cannot raise queue.Full again because we are the only ones adding items to the queue, and we either just removed one or the queue was magically emptied before we got to it
		return None, pyaudio.paContinue

# Speech detector class
class SpeechDetector:

	chunk_history: collections.deque[bytes]
	chunk_id: int
	state: DetectorState
	init_lockout: int
	loud_start_count: int
	silent_start_count: int
	silent_stop_count: int
	starting_padding: int
	starting_id: int
	stopped_id: int

	def __init__(self, chunk_time, sample_width, energy_silence=65, energy_start=200, energy_stop=120, energy_dynamic=True, energy_dynamic_ts90=1.2, starting_duration=0.125, start_duration=0.25, stop_duration=1.2, padding_duration=0.7, debug=False):
		# chunk_time = Audio time corresponding to each chunk (assumed constant)
		# sample_width = Width in bytes of each sample within a chunk
		# energy_silence = Assumed energy of non-speech audio (only used if dynamic energy threshold is enabled, in order to scale target thresholds relative to measured silence energy)
		# energy_start = Energy threshold to allow start of speech to be detected
		# energy_stop = Energy threshold to allow stop of speech to be detected
		# energy_dynamic = Whether to dynamically adjust the energy thresholds to try to adapt to measured silence energy levels
		# energy_dynamic_ts90 = 90% settling time in seconds of the dynamic energy threshold adjustment
		# starting_duration = Amount of consistent audio above the start threshold that triggers retrospective starting, i.e. how many of the first consecutive start chunks must be above the start energy threshold (quantised to nearest chunk)
		# start_duration = Minimum amount of speech in order to retrospectively trigger start of recording (quantised to nearest chunk)
		# stop_duration = Amount of silence in order to trigger stop recording (quantised to nearest chunk)
		# padding_duration = Maximum amount of silence to keep in the recording before and after actual speech (quantised to nearest chunk)
		# debug = Whether to show output for debugging

		self.chunk_time = chunk_time
		self.sample_width = sample_width
		self.energy_silence = energy_silence
		self.energy_start = energy_start
		self.energy_stop = energy_stop
		self.energy_dynamic = energy_dynamic
		self.energy_dynamic_ts90 = energy_dynamic_ts90
		self.starting_duration = starting_duration
		self.start_duration = start_duration
		self.stop_duration = stop_duration
		self.padding_duration = padding_duration
		self.debug = debug

		if self.energy_dynamic_ts90 < self.chunk_time:
			rospy.logwarn(f"Dynamic energy threshold 90% settling time ({self.energy_dynamic_ts90:.3g}) must be at least the chunk time ({self.chunk_time:.3g}) => Increasing settling time")
			self.energy_dynamic_ts90 = self.chunk_time
		if self.starting_duration > self.start_duration:
			rospy.logwarn(f"Starting duration {self.starting_duration:.3g} must be less than start duration {self.start_duration:.3g} => Increasing start duration")
			self.start_duration = self.starting_duration
		if self.padding_duration > self.stop_duration:
			rospy.logwarn(f"Padding duration {self.padding_duration:.3g} must be less than stop duration {self.stop_duration:.3g} => Reducing padding duration")
			self.padding_duration = self.stop_duration

		self.energy_start_ratio = self.energy_start / self.energy_silence
		self.energy_stop_ratio = self.energy_stop / self.energy_silence
		if self.energy_start_ratio <= 1 or self.energy_stop_ratio <= 1 or self.energy_stop > self.energy_start:
			raise ValueError(f"Must have energy silence < stop <= start ({self.energy_silence:.1f} < {self.energy_stop:.1f} <= {self.energy_start:.1f})")

		self.num_ts90_chunks = max(round(self.energy_dynamic_ts90 / self.chunk_time), 1)
		self.num_starting_chunks = max(round(self.starting_duration / self.chunk_time), 1)
		self.num_start_chunks = max(round(self.start_duration / self.chunk_time), 1)
		self.num_stop_chunks = max(round(self.stop_duration / self.chunk_time), 1)
		self.num_padding_chunks = max(round(self.padding_duration / self.chunk_time), 1)

		silence_energy_factor_inv = 0.10 ** (self.chunk_time / self.energy_dynamic_ts90)
		self.silence_energy_factor = 1 - silence_energy_factor_inv

		rospy.loginfo(f"Speech detector has min {self.num_starting_chunks * self.chunk_time:.3g}s/{self.num_start_chunks * self.chunk_time:.3g}s speech, {self.num_stop_chunks * self.chunk_time:.3g}s stop silence, {self.num_padding_chunks * self.chunk_time:.3g}s padding")
		rospy.loginfo(f"Speech energy thresholds are {self.energy_silence:.0f} silence, {self.energy_start:.0f} start, {self.energy_stop:.0f} stop{f' (dynamic Ts90 of {self.energy_dynamic_ts90:.3g}s)' if self.energy_dynamic else ''}")
		if self.debug:
			rospy.loginfo(f"Padding {self.num_padding_chunks}, starting {self.num_starting_chunks}/{self.num_start_chunks} min speech chunks, stop silence {self.num_stop_chunks}{f', dynamic Ts90 {self.num_ts90_chunks}' if self.energy_dynamic else ''}")

		self.chunk_history = collections.deque(maxlen=self.num_padding_chunks + self.num_start_chunks + self.num_stop_chunks - 1)
		self.reset()

	def reset(self, lockout_duration=0.0):
		# Note: This does not reset self.energy_silence, self.energy_start, self.energy_stop so that silence threshold estimation transcends individual recognition runs
		self.chunk_history.clear()       # Rolling history window of latest chunks
		self.chunk_id = -1               # ID of the latest chunk in the history (0-indexed)
		self.state = DetectorState.INIT  # Current speech detection state
		self.init_lockout = math.ceil(lockout_duration / self.chunk_time)  # Minimum number of initial chunks to prevent a transition out of the INIT state for
		self.loud_start_count = 0        # Number of latest consecutive chunks above the start energy threshold
		self.silent_start_count = 0      # Number of consecutive chunks below the start energy threshold prior to any current consecutive series of chunks above the start energy threshold
		self.silent_stop_count = 0       # Number of latest consecutive chunks below the stop energy threshold
		self.starting_padding = 0        # If in the STARTING state, the amount of available silence padding prior to starting
		self.starting_id = -1            # The chunk ID right before the first loud one, or the earliest chunk ID that can possibly still become that
		self.stopped_id = -1             # If in the STOPPED state, the last padding STARTED chunk immediately after which the STOPPED state was entered

	def process_chunk(self, chunk: bytes) -> tuple[int, float, float, float, float, int, DetectorState, int]:
		# Returns (start/stopping of speech is never detected in the same chunk):
		#  - Chunk ID of input chunk
		#  - Energy of input chunk
		#  - Silence energy level used when processing input chunk
		#  - Start energy threshold used when processing input chunk
		#  - Stop energy threshold used when processing input chunk
		#  - After processing the input chunk, the new chunk ID right before the first loud one, or the earliest chunk ID that can possibly still become that
		#  - Current state of the detector
		#  - How many chunks (including input chunk) need to be trimmed if stopping a currently running recording

		self.chunk_id += 1
		self.chunk_history.append(chunk)
		chunk_energy = audioop.rms(chunk, self.sample_width)

		if self.state == DetectorState.INIT:

			if chunk_energy > 0:
				if chunk_energy <= self.energy_stop:
					self.silent_stop_count += 1
					if self.silent_stop_count >= self.num_starting_chunks and self.chunk_id >= self.init_lockout - 1:
						self.silent_stop_count = 0
						self.stopped_id = self.chunk_id
						self.state = DetectorState.STOPPED
						if self.debug:
							rospy.loginfo(f"UNLOCKED after chunk {self.chunk_id} (last ignored chunk)")
				elif self.silent_stop_count < self.num_starting_chunks:
					self.silent_stop_count = 0
			self.starting_id = self.chunk_id

		else:

			if chunk_loud_start := (chunk_energy > self.energy_start):
				self.loud_start_count += 1
			else:
				if self.loud_start_count > 0:
					self.silent_start_count = 1
				else:
					self.silent_start_count += 1
				self.loud_start_count = 0
				if self.state == DetectorState.STOPPED:
					self.starting_id = self.chunk_id

			if chunk_energy <= self.energy_stop:
				self.silent_stop_count += 1
			else:
				self.silent_stop_count = 0

			if self.state == DetectorState.STOPPED and self.loud_start_count >= self.num_starting_chunks:
				self.starting_padding = min(self.silent_start_count, self.num_padding_chunks)
				self.state = DetectorState.STARTING
				if self.debug:
					rospy.loginfo(f"STARTING at chunk {self.chunk_id} with start energy threshold {self.energy_start:.0f}, first {self.starting_id - self.starting_padding + 1}, pre-loud {self.starting_id}, num pre-padding {self.starting_padding}")

			if self.state == DetectorState.STARTING:
				if self.silent_stop_count >= self.num_stop_chunks:
					self.state = DetectorState.STOPPED
					self.starting_id = self.chunk_id
					if self.debug:
						rospy.loginfo(f"ABORTING at chunk {self.chunk_id} with stop energy threshold {self.energy_stop:.0f} due to {self.silent_stop_count} stop silence chunks")
				elif chunk_loud_start and self.chunk_id - self.starting_id >= self.num_start_chunks:
					self.state = DetectorState.STARTED
					if self.debug:
						rospy.loginfo(f"STARTED at chunk {self.chunk_id} with initial {self.starting_padding} pre-padding and {self.chunk_id - self.starting_id} speech chunks")
			elif self.state == DetectorState.STARTED:
				if self.silent_stop_count >= self.num_stop_chunks:
					self.state = DetectorState.STOPPED
					self.starting_id = self.chunk_id
					self.stopped_id = self.chunk_id - self.silent_stop_count + self.num_padding_chunks  # <= self.chunk_id
					if self.debug:
						rospy.loginfo(f"STOPPED at chunk {self.chunk_id} with stop energy threshold {self.energy_stop:.0f} with last {self.stopped_id}, {self.chunk_id - self.stopped_id} trimmed silence chunks")

		last_energy_silence = self.energy_silence
		last_energy_start = self.energy_start
		last_energy_stop = self.energy_stop
		if self.energy_dynamic and chunk_energy > 0 and (self.state == DetectorState.INIT or (self.state == DetectorState.STOPPED and chunk_energy <= self.energy_start)):
			self.energy_silence += self.silence_energy_factor * (chunk_energy - self.energy_silence)
			self.energy_start = self.energy_start_ratio * self.energy_silence
			self.energy_stop = self.energy_stop_ratio * self.energy_silence

		trim_stop_chunks = self.chunk_id - self.stopped_id
		return self.chunk_id, chunk_energy, last_energy_silence, last_energy_start, last_energy_stop, self.starting_id, self.state, trim_stop_chunks

	def can_start(self, first_id: int) -> Optional[list[bytes]]:
		# first_id = The earliest chunk ID that is allowed to be in a particular recording
		# If the current started speech can trigger a recording while respecting the given first_id, then returns the corresponding previous recorded frames (last frame is the one before current chunk ID), otherwise returns None
		if self.state == DetectorState.STARTED:
			pre_first_id = first_id - 1
			if self.chunk_id - self.silent_stop_count - max(pre_first_id, self.starting_id) >= self.num_start_chunks:
				num_started_chunks = self.chunk_id - max(pre_first_id, self.starting_id - self.starting_padding)  # 1 <= self.num_start_chunks + self.silent_stop_count <= self.chunk_id - max(pre_first_id, self.starting_id) <= num_started_chunks <= self.chunk_id - self.starting_id + self.starting_padding <= self.num_padding_chunks + self.num_start_chunks + self.num_stop_chunks - 1 == self.chunk_history.maxlen
				previous_chunks = list(itertools.islice(self.chunk_history, len(self.chunk_history) - num_started_chunks, len(self.chunk_history) - 1))
				if self.debug:
					rospy.loginfo(f"CAN START at chunk {self.chunk_id} with first {self.chunk_id - num_started_chunks + 1} and {num_started_chunks} total chunks")
				return previous_chunks
		return None

# Delayed item delete context manager for dicts
class DelayedDeleter:

	def __init__(self, dic: dict):
		self.dic = dic
		self.delete_keys = None

	def delete_now(self, key):
		# Use this outside of context manager
		del self.dic[key]

	def __enter__(self):
		self.delete_keys = None
		return self

	def delete(self, key):
		# Use this within context manager
		if self.delete_keys is None:
			self.delete_keys = [key]
		else:
			self.delete_keys.append(key)

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self.delete_keys is not None:
			for key in self.delete_keys:
				del self.dic[key]
			self.delete_keys = None
		return False

# Run main function
if __name__ == "__main__":
	main()
# EOF
