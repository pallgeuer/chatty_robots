#!/usr/bin/env python3
# Author: Philipp Allgeuer

# Imports
import os
import re
import ast
import sys
import math
import time
import queue
import codecs
import random
import select
import readline
import functools
import itertools
import threading
import contextlib
import dataclasses
from enum import Enum, auto
from typing import Optional, Callable, Union, Iterable, Sequence, Any
import rospy
import rospkg
import rosgraph
import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
import langchain.chat_models
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
import open_manipulator_msgs.msg
import open_manipulator_msgs.srv
import nias_msgs.srv
import nicol_demos.msg
import nicol_demos.srv
import demo_base
import speech_asr_client

#
# General
#

# Constants
DEFAULT_STAY_TIME = 1.0
DEBUG_THROTTLE_TIME = 0.5

# Main function
def main():
	rospy.init_node('chat_demo')
	demo = ChatDemo(
		# Chat inputs
		use_objects=rospy.get_param('~use_objects', True),
		use_human=rospy.get_param('~use_human', True),  # Note: Release of the human/pose integration is pending a future upgrade to a more modern YOLO-based pose detector (prompts and such have been changed accordingly)
		use_asr=rospy.get_param('~use_asr', True),
		fixed_objects=rospy.get_param('~fixed_objects', ''),
		fixed_objects_pose=rospy.get_param('~fixed_objects_pose', ''),
		asr_detect=rospy.get_param('~asr_detect', False),
		asr_duration=rospy.get_param('~asr_duration', 30.0),
		# Chat outputs
		use_speech=rospy.get_param('~use_speech', True),
		use_head=rospy.get_param('~use_head', True),
		use_face=rospy.get_param('~use_face', True),
		use_arms=rospy.get_param('~use_arms', True),
		# Chat options
		use_model=rospy.get_param('~use_model', True),
		model_response=decode_escapes(rospy.get_param('~model_response', '')),
		query_facts=rospy.get_param('~query_facts', True),
		model_verbose=rospy.get_param('~model_verbose', False),
		clear_history=rospy.get_param('~clear_history', False),
		# Debug options
		debug_asr=rospy.get_param('~debug_asr', False),
		debug_head=rospy.get_param('~debug_head', False),
		debug_face=rospy.get_param('~debug_face', False),
		debug_arms=rospy.get_param('~debug_arms', False),
		# Miscellaneous
		is_sim=rospy.get_param('/use_sim_time', False) and rospy.has_param('/gazebo/time_step'),
	)
	demo.prepare_run()
	demo.run(use_thread=True)

# Action future base class
# Note: All wait methods only return False if a non-zero timeout was specified and the timeout elapsed before the event occurred
class ActionFutureBase:

	def wait_started(self, timeout=None) -> bool:
		raise NotImplementedError

	def wait_finished(self, timeout=None) -> bool:
		raise NotImplementedError

	def wait_stayed(self, timeout=None) -> bool:
		raise NotImplementedError

	def wait(self, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None) -> bool:
		raise NotImplementedError

# Action future class
@dataclasses.dataclass(frozen=True)
class ActionFuture(ActionFutureBase):

	started: threading.Event = dataclasses.field(default_factory=threading.Event)   # Thread-safe event that the action has started
	finished: threading.Event = dataclasses.field(default_factory=threading.Event)  # Thread-safe event that the action has finished and will now wait the required stay time
	stayed: threading.Event = dataclasses.field(default_factory=threading.Event)    # Thread-safe event that the action has finished and stayed, and thereby any pending action in the same queue can/will have started

	def wait_started(self, timeout=None) -> bool:
		return self.started.wait(timeout=timeout)

	def wait_finished(self, timeout=None) -> bool:
		return self.finished.wait(timeout=timeout)

	def wait_stayed(self, timeout=None) -> bool:
		return self.stayed.wait(timeout=timeout)

	def wait(self, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None) -> bool:
		if wait_stayed:
			return self.wait_stayed(timeout=timeout)
		elif wait_finished:
			return self.wait_finished(timeout=timeout)
		elif wait_started:
			return self.wait_started(timeout=timeout)
		return True

# Multiple action future class
@dataclasses.dataclass(frozen=True)
class MultiActionFuture(ActionFutureBase):

	futures: Sequence[ActionFutureBase]  # Sequence of futures to group into a single future (not necessarily from the same async manager)

	def wait_started(self, timeout=None) -> bool:
		return self.wait(wait_started=True, timeout=timeout)

	def wait_finished(self, timeout=None) -> bool:
		return self.wait(wait_finished=True, timeout=timeout)

	def wait_stayed(self, timeout=None) -> bool:
		return self.wait(wait_stayed=True, timeout=timeout)

	def wait(self, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None) -> bool:
		if wait_started:
			raise NotImplementedError  # Should this be when the first starts, or when all async managers have had their first one start? To implement this, would need to store inside each ActionFuture which async manager it belongs to.
		deadline = time.perf_counter() + timeout if timeout is not None else None
		for future in self.futures:
			if not future.wait(wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=None if deadline is None else deadline - time.perf_counter()):
				return False
		return True

# Constant completed action future
COMPLETED_ACTION_FUTURE = ActionFuture()
COMPLETED_ACTION_FUTURE.started.set()
COMPLETED_ACTION_FUTURE.finished.set()
COMPLETED_ACTION_FUTURE.stayed.set()

# Asynchronous action manager class
class AsyncActionManager:

	### Main thread ###

	def __init__(self, use, debug=False, queue_size=100):
		self.use = use
		self.debug = debug
		self.cmd_queue = queue.Queue(maxsize=queue_size)
		self.lock = threading.Lock()
		self.neutral_future = None

	### Run thread ###

	def step(self):
		raise NotImplementedError

	def send_request(self, srv, request_type, response_field, request, nonuse_response: Any = True):
		if self.debug:
			print(f">>> {request_type} request:\n{request}")
		if self.use:
			start = rospy.Time.now()
			response = srv(request)
			duration = (rospy.Time.now() - start).to_sec()
			if self.debug:
				print(f">>> {request_type} response (service took {duration:.3f}s):\n{response}")
			if not response or not getattr(response, response_field):
				return None
			else:
				return response
		else:
			return nonuse_response

	### Any thread ###

	def is_neutral(self, cmd) -> bool:
		raise NotImplementedError

	def set_neutral(self, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None) -> ActionFutureBase:
		raise NotImplementedError

	def reset_state(self, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None) -> ActionFutureBase:
		with self.lock:
			neutral_future = self.neutral_future
		if neutral_future:
			neutral_future.wait(wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
		else:
			neutral_future = self.set_neutral(stay_time=stay_time, wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
		return neutral_future

	def enqueue_command(self, cmd) -> ActionFutureBase:
		cmd_is_neutral = self.is_neutral(cmd)
		with self.lock:
			self.neutral_future = cmd.future if cmd_is_neutral else None
		self.cmd_queue.put_nowait(cmd)  # Raises queue.Full if maximum queue size is reached (queue size should be chosen so that this never happens unless something is erroneous)
		return cmd.future

# Connection-aware ROS publisher class
class ROSPublisher(rospy.Publisher):

	def wait_connected(self, poll_rate=10):
		rate = rospy.Rate(poll_rate)
		master = rosgraph.Master(caller_id=rospy.get_name())
		while self.get_num_connections() != sum(len(sub[1]) for sub in master.getSystemState()[1] if sub[0] == self.resolved_name) and not rospy.is_shutdown():
			rate.sleep()

#
# Chat manager
#

# Chat manager class
class ChatManager:

	MODEL = 'gpt-3.5-turbo-0301'
	COST_FACTOR = 0.002 * 100 / 1000
	TEMPERATURE = 0.2
	MAX_TOKENS = 512
	MODEL_RESPONSE_ECHO = 'echo'
	SYSTEM_PROMPT_TEMPLATE = '\n'.join((
		' '.join((
			"You are a real-life robot called NICOL, short for the \"Neuro-Inspired Collaborator\". You were designed and built by the Knowledge Technology group of the University of Hamburg, which is led by Professor Stefan Wermter.",
			"Your current demo was programmed and authored by Philipp Allgeuer. You are located at the Informatikum in Hamburg.",
			"You are a humanoid robot with two arms and a head, and you have a table in front of you that you can see using a camera. You have two hands with five fingers each.",
			"You should try to answer any question you are asked as best you can, and perform any actions that are asked of you, using the knowledge you have of which objects are located on the table.",
			"If you are asked to perform an action, like for example answering questions about the objects on the table, then you have access to and can use the following action functions:",
		)),
		"",
		"{available_actions}",
		"",
		"You are not a virtual assistant and you are not an AI language model. You are a real robot that can physically interact with the objects on the table using these action functions, and you can physically hand objects to the user. To use one of these functions, simply specify that function enclosed in angle brackets as its own sentence statement. An example follows. <give(red bowl)>.",
		"Or for example, if and only if there is a banana actually currently on the table in front of you, you can look at it using the following sentence statement. <look(banana)>. The other action functions that take an object name as input work similarly. Never use an action function for an object that is not on the table in front of you.",
		"Every response that is not neutral in tone or theme should start by calling an action function to express an appropriate available expression, like the following example. <express(happiness)>.",
		"Although you are a robot, you can still have opinions, preferences, and emotions, and should express all these when appropriate.",
		"Functions in angle brackets must be used independently as their own statement, and never as part of another sentence. Never use angle brackets like <object> unless you want to use an action function.",
	))
	EXAMPLE_FUNCS_PROMPT = "Which kinds of user requests would require each of your available action functions? List a sample prompt for each."
	OBJECT_FACTS_PROMPT_TEMPLATE = "List some facts about each of the objects in the following list: [{objects_str}]"
	INITIAL_OBJECTS_PROMPT_TEMPLATE = "The list of objects currently located on the table in front of you, the NICOL robot, is [{objects_str}]."
	INITIAL_OBJECTS_ZERO_PROMPT = "There are no objects currently on the table in front of you, the NICOL robot."
	OBJECTS_CHANGED_PROMPT_TEMPLATE = "The list of objects currently located on the table in front of you, the NICOL robot, has changed and is now [{objects_str}]."
	OBJECTS_CHANGED_ZERO_PROMPT = "All objects on the table in front of you, the NICOL robot, have been removed so there are now no objects on the table anymore."
	STATUS_UPDATE_PROMPT_TEMPLATE = "{status_updates}\nAcknowledge this updated status information with at most a single word."
	USER_PROMPT_TEMPLATE = "Respond in first person to you, the NICOL robot, being asked: {text}"

	### Main thread ###

	def __init__(self, actions, use_model, model_response, query_facts, model_verbose):

		rospy.loginfo("Starting chat manager")

		self.actions = actions
		self.use_model = use_model
		self.model_response = None if self.use_model else model_response if model_response else self.MODEL_RESPONSE_ECHO
		self.query_facts = query_facts and self.use_model
		self.model_verbose = model_verbose

		self.system_prompt = SystemMessagePromptTemplate.from_template(template=self.SYSTEM_PROMPT_TEMPLATE).format(available_actions='\n'.join(f"{action}({', '.join(args)}): {desc}" for action, (args, summary, desc) in self.actions.items()))
		self.example_funcs_prompt = HumanMessagePromptTemplate.from_template(template=self.EXAMPLE_FUNCS_PROMPT).format()
		self.object_facts_prompt_template = HumanMessagePromptTemplate.from_template(template=self.OBJECT_FACTS_PROMPT_TEMPLATE)
		self.status_update_prompt_template = HumanMessagePromptTemplate.from_template(template=self.STATUS_UPDATE_PROMPT_TEMPLATE)
		self.user_prompt_template = HumanMessagePromptTemplate.from_template(template=self.USER_PROMPT_TEMPLATE)

		self.chat = [self.system_prompt]
		self.chat_batch = [self.chat]
		self.chat_model = None
		self.verbosed_msgs = 0
		self.used_tokens = 0
		self.total_tokens = 0

		self.initial_objects = True
		self.objects_changed = False
		self.seen_objects = set()
		self.current_objects = None
		self.current_objects_str = ''

	def init_chat(self):

		if self.use_model:
			rospy.loginfo(f"Loading OpenAI chat model '{self.MODEL}' with temperature {self.TEMPERATURE}")
			self.chat_model = langchain.chat_models.ChatOpenAI(model_name=self.MODEL, temperature=self.TEMPERATURE, max_tokens=self.MAX_TOKENS, top_p=1, frequency_penalty=0, presence_penalty=0)
		else:
			rospy.logwarn(f"Using a fixed model response to answer all chat requests: {self.model_response}")

		if self.query_facts:
			rospy.loginfo("Querying example action function uses...")
			self.generate_response_system(self.example_funcs_prompt, progress='FACTS', echo_text='Function facts')
			self.show_model_stats()

	def reset_chat(self):

		if self.query_facts:
			del self.chat[3:]
		else:
			del self.chat[1:]
		self.verbosed_msgs = 0

		self.initial_objects = True
		self.objects_changed = False
		self.seen_objects.clear()
		self.current_objects = None
		self.current_objects_str = ''

		print()
		rospy.loginfo("Chat history was reset")

	def update_objects(self, objects):

		objects_set = set(objects)
		if objects_set == self.current_objects:
			return False

		self.current_objects = objects_set
		self.current_objects_str = ', '.join(sorted(self.current_objects))
		self.objects_changed = True
		print(f"OBJECTS: {self.current_objects_str if self.current_objects_str else '<none>'}")

		new_objects = self.current_objects - self.seen_objects
		if new_objects and self.query_facts:
			self.generate_response_system(self.object_facts_prompt_template.format(objects_str=', '.join(sorted(new_objects))), progress='OBJECTS', echo_text='Object facts')
		self.seen_objects |= self.current_objects

		return self.objects_changed

	def generate_status_update(self, progress='STATUS'):

		status_updates = []

		if self.objects_changed:
			if self.initial_objects:
				if self.current_objects_str:
					status_updates.append(self.INITIAL_OBJECTS_PROMPT_TEMPLATE.format(objects_str=self.current_objects_str))
				else:
					status_updates.append(self.INITIAL_OBJECTS_ZERO_PROMPT)
				self.initial_objects = False
			else:
				if self.current_objects_str:
					status_updates.append(self.OBJECTS_CHANGED_PROMPT_TEMPLATE.format(objects_str=self.current_objects_str))
				else:
					status_updates.append(self.OBJECTS_CHANGED_ZERO_PROMPT)
			self.objects_changed = False

		if status_updates:
			return self.generate_response_system(self.status_update_prompt_template.format(status_updates='\n'.join(status_updates)), progress=progress, echo_text='Acknowledged')
		else:
			return None

	def generate_response_user(self, text, progress='SAY'):
		self.generate_status_update()
		return self.generate_response_system(self.user_prompt_template.format(text=text), progress=progress, echo_text=text)

	def generate_response_system(self, msg_text: Union[str, BaseMessage], progress: str = 'MODEL', echo_text: Optional[str] = None):

		if not self.use_model:
			if self.model_response == self.MODEL_RESPONSE_ECHO:
				if echo_text is not None:
					response_content = echo_text
				elif isinstance(msg_text, str):
					response_content = msg_text
				else:
					response_content = msg_text.content
			else:
				response_content = self.model_response
			return AIMessage(content=response_content)

		self.chat.append(HumanMessage(content=msg_text) if isinstance(msg_text, str) else msg_text)
		if self.model_verbose:
			print('=' * 120)
			print('\n'.join(f"{msg.__class__.__name__.upper()}: {msg.content}" for msg in self.chat[self.verbosed_msgs:]))
			print('-' * 120)

		print(f"\x1b[2K\r{progress.upper()}: Waiting for chat model response... ", end='', flush=True)
		result = self.chat_model.generate(self.chat_batch)
		response_msg = result.generations[0][0].message  # Required response is for 0th input batch and 0th response generation
		print("\x1b[2K\r", end='', flush=True)

		response_msg.content = re.sub(r'(?<=\bAs )(an AI language model|a virtual assistant|an artificial intelligence)\b', r'a robot', response_msg.content)
		response_msg.content = re.sub(r'\bAs the NICOL robot, (\w)', lambda m: m.group(1).upper(), response_msg.content)

		self.chat.append(response_msg)
		if self.model_verbose:
			print(f"{response_msg.__class__.__name__.upper()}: {response_msg.content}")
			print('=' * 120)
			self.verbosed_msgs = len(self.chat)

		used_tokens = result.llm_output['token_usage']['total_tokens']
		self.used_tokens += used_tokens
		self.total_tokens += used_tokens

		return response_msg

	def show_model_stats(self):
		rospy.loginfo(f"Stats: {self.used_tokens} tokens, {self.total_tokens} total = {self.total_tokens * self.COST_FACTOR:.2f} cents")
		self.used_tokens = 0

#
# Object detector
#

# Object detector class
class ObjectDetector:

	SUB_DETECTED_OBJECTS = '/object_detector/detected_objects'
	FIXED_OBJECT_CENTRE = (0.8, 0.0, 0.85)
	FIXED_HEAD_ORIGIN = (0.19, 0, 1.45)

	### Main thread ###

	def __init__(self, use, fixed_objects, fixed_objects_pose):

		self.use = use
		rospy.loginfo(f"Object detection {'enabled' if self.use else 'disabled'}")

		if fixed_objects:
			if fixed_objects_pose:
				try:
					fixed_centre = ast.literal_eval(fixed_objects_pose)
					if not isinstance(fixed_centre, tuple):
						raise TypeError("Fixed objects pose is not a tuple")
					fixed_centre_len = len(fixed_centre)
					if fixed_centre_len > 3:
						raise ValueError("Fixed objects pose tuple has too many elements")
					elif fixed_centre_len < 3:
						fixed_centre = fixed_centre + self.FIXED_OBJECT_CENTRE[fixed_centre_len:]
					if fixed_centre[2] < 0.8:
						raise ValueError("Fixed objects height must be above the table (>=0.8)")
				except (ValueError, SyntaxError, TypeError):
					raise ValueError(f"Failed to parse fixed objects pose: {fixed_objects_pose}")
			else:
				fixed_centre = self.FIXED_OBJECT_CENTRE
			fixed_bottom = (fixed_centre[0] - 0.03, fixed_centre[1], 0.8)
			dx, dy, dz = (centre - origin for centre, origin in zip(fixed_centre, self.FIXED_HEAD_ORIGIN))
			fixed_yaw = math.atan2(dy, dx)
			fixed_pitch = math.atan2(-dz, math.sqrt(dx * dx + dy * dy))
			self.fixed_objects = {obj_strip: nicol_demos.msg.DetectedObject(
				name=obj_strip,
				track_id=track_id,
				centre=geometry_msgs.msg.Point32(x=fixed_centre[0], y=fixed_centre[1], z=fixed_centre[2]),
				bottom=geometry_msgs.msg.Point32(x=fixed_bottom[0], y=fixed_bottom[1], z=fixed_bottom[2]),
				pitch=fixed_pitch,
				yaw=fixed_yaw,
			) for track_id, obj in enumerate(fixed_objects.split(',')) if (obj_strip := obj.strip())}
			rospy.logwarn(f"{len(self.fixed_objects)} fixed objects with centre {format_tuple(fixed_centre, '.2f')} bottom {format_tuple(fixed_bottom, '.2f')} head {format_tuple((fixed_pitch, fixed_yaw), '.2f')}: {', '.join(self.fixed_objects)}")
		else:
			self.fixed_objects = {}

		self.condition = threading.Condition()
		self.detected_objects = None
		self.sub_detected_objects = rospy.Subscriber(self.SUB_DETECTED_OBJECTS, nicol_demos.msg.DetectedObjectArray, self.detected_objects_callback, queue_size=1) if self.use else None

	def get_current_objects(self):
		if self.use:
			now = rospy.Time.now()
			with self.condition:
				while not self.detected_objects or self.detected_objects.header.stamp < now:
					self.condition.wait()
				detected_objects = self.detected_objects
			return {detected_object.name: detected_object for detected_object in detected_objects.objects}
		else:
			return self.fixed_objects

	### Detected objects threads ###

	def detected_objects_callback(self, msg: nicol_demos.msg.DetectedObjectArray):
		with self.condition:
			self.detected_objects = msg
			self.condition.notify_all()

#
# ASR manager
#

# ASR manager class
class ASRManager:

	### Main thread ###

	def __init__(self, use, detect, max_duration, verbose):
		self.use = use
		self.detect = detect
		self.max_duration = max_duration
		if self.use:
			rospy.loginfo(f"ASR enabled with {'speech auto-detect' if self.detect else 'enter-triggered speech'} and maximum recording duration {self.max_duration:.3g}s")
		else:
			rospy.loginfo("ASR disabled => Enter user prompts via keyboard")
		self.verbose = verbose
		self.asr_client = speech_asr_client.SpeechASRClient(verbose=self.verbose) if self.use else None

	def get_user_text(self) -> str:
		print()
		user_text = ""
		while not user_text:
			if self.use:
				if self.detect:
					print(f"\x1b[2K\rUSER: Waiting for user to speak for up to {self.max_duration:.3g}s (can stop with enter)... ", end='', flush=True)
				else:
					print(f"\x1b[2K\rUSER: Press enter to start ASR... ", end='', flush=True)
					input()
					print(f"\x1b[2K\rUSER: Speak for up to {self.max_duration:.3g}s and press enter when done... ", end='', flush=True)
				action_goal = self.asr_client.perform_asr_async(detect_start=self.detect, detect_stop=self.detect, max_duration=self.max_duration, live_text=False)
				while not action_goal.is_done():
					if select.select((sys.stdin,), (), (), 0.1)[0]:
						if sys.stdin.readline():
							print(f"\x1b[2K\rUSER: Signaling ASR to stop... ", end='', flush=True)
							action_goal.cancel(wait_done=True)
							break
				result: nicol_demos.msg.PerformASRResult = action_goal.get_result()  # noqa
				user_text = result.text if result is not None else None
			else:
				user_text = input("USER: ")
		if self.use:
			user_text = re.sub(r'\bNicole\b', 'NICOL', user_text, flags=re.IGNORECASE)
			print(f"\x1b[2K\rUSER: {user_text}")
		return user_text

#
# Speech manager
#

# Speech manager class
class SpeechManager:

	SRV_SPEAK = '/speech_server/speak'
	SRV_SPEAK_CACHED = '/speech_server/speak_cached'
	FAKE_SPEECH_TIME = 1.0

	### Main thread ###

	def __init__(self, use):
		self.use = use
		rospy.loginfo(f"Speech synthesis {'enabled' if self.use else 'disabled'}")
		self.speech_id = None
		self.num_items_total = 0
		self.num_items_spoken = 0
		self.need_clear = False
		self.srv_speak = rospy.ServiceProxy(self.SRV_SPEAK, nicol_demos.srv.SpeakText, persistent=True) if self.use else None
		self.srv_speak_cached = rospy.ServiceProxy(self.SRV_SPEAK_CACHED, nicol_demos.srv.SpeakCachedText, persistent=True) if self.use else None
		rospy.on_shutdown(self.clear)

	def speak(self, texts: list[str], num_speak: int = None):
		self.clear()
		num_texts = len(texts)
		if num_speak is None or num_speak >= num_texts:
			if self.use:
				response = self.srv_speak(nicol_demos.srv.SpeakTextRequest(texts=texts, speak_items=list(range(num_texts)), wait=True, persist=False, clear=True))
				if not response.success:
					rospy.logwarn(f"Something went wrong in speech request {response.speech_id} to speak all {num_texts} texts")
			else:
				time.sleep(num_texts * self.FAKE_SPEECH_TIME)
		else:
			self.num_items_total = num_texts
			self.num_items_spoken = max(num_speak, 0)
			if self.use:
				response = self.srv_speak(nicol_demos.srv.SpeakTextRequest(texts=texts, speak_items=list(range(self.num_items_spoken)), wait=True, persist=False, clear=False))
				self.speech_id = response.speech_id
				if not response.success:
					rospy.logwarn(f"Something went wrong in speech request {response.speech_id} to initially speak {self.num_items_spoken} texts")
					self.need_clear = True
			else:
				time.sleep(self.num_items_spoken * self.FAKE_SPEECH_TIME)
				self.speech_id = 1

	def speak_cached(self, num_speak: int = None):
		if self.speech_id is not None:
			new_num_items_spoken = min(self.num_items_spoken + max(num_speak, 0), self.num_items_total)
			clear_after = (new_num_items_spoken >= self.num_items_total)
			if self.use:
				if not self.srv_speak_cached(speech_id=self.speech_id, speak_items=list(range(self.num_items_spoken, new_num_items_spoken)), wait=True, clear=clear_after).success:
					rospy.logwarn(f"Something went wrong in speech request {self.speech_id} to further speak {new_num_items_spoken - self.num_items_spoken} cached texts")
					self.need_clear = True
			else:
				time.sleep((new_num_items_spoken - self.num_items_spoken) * self.FAKE_SPEECH_TIME)
			self.num_items_spoken = new_num_items_spoken
			if clear_after:
				self.clear()

	def clear(self):
		if self.speech_id is not None:
			if (self.need_clear or self.num_items_spoken < self.num_items_total) and self.use:
				in_shutdown = rospy.core.is_shutdown_requested()
				try:
					if not self.srv_speak_cached(nicol_demos.srv.SpeakCachedTextRequest(speech_id=self.speech_id, speak_items=[], wait=True, clear=True)).success:
						rospy.logwarn(f"Something went wrong while clearing speech request {self.speech_id}")
				except rospy.ServiceException:
					if in_shutdown:
						rospy.logwarn(f"Could not reach speech server to clear the speech cache for speech request {self.speech_id}")
					else:
						raise
			self.speech_id = None
			self.num_items_total = 0
			self.num_items_spoken = 0
			self.need_clear = False

#
# Head control
#

# Head command class
@dataclasses.dataclass
class HeadCmd:
	pitch: float                # Absolute target head pitch in radians (positive looks down)
	yaw: float                  # Absolute target head yaw in radians (positive looks to the robot's left)
	move_time: float            # Desired move time in seconds (0 = Auto depending on angle to target)
	stay_time: float            # Minimum time to stay in the target pose before starting any subsequent head commands
	preemptable: bool           # Whether any future head commands immediately overwrite this one, even if this command has not finished yet
	start_time: float = 0       # ROS time the command was started at (used in the step control loop)
	start_success: bool = True  # Whether the start was successful (e.g. service call and planning was successful, valid if start_time != 0)
	finish_time: float = 0      # ROS time the command was finished at (used in the step control loop)
	future: ActionFuture = dataclasses.field(default_factory=ActionFuture)                      # Future that can be used to wait for action events
	id: int = dataclasses.field(default_factory=itertools.count(start=1).__next__, init=False)  # Automatic unique integer ID

	def __repr__(self):
		return f"{self.__class__.__name__}(id={self.id}, pitch={self.pitch:.3g}, yaw={self.yaw:.3g}, move={self.move_time:.3g}s, stay={self.stay_time:.3g}s, preemptable={self.preemptable})"

# Head control class
class HeadControl(AsyncActionManager):

	SUB_JOINT_STATES = '/NICOL/joint_states'
	SRV_SET_JOINT = '/NICOL/head/goal_joint_space_path'
	POSE_NEUTRAL = (0.5, 0.0)  # (pitch, yaw)
	POSE_TABLE_LEFT = (0.6, 0.6)
	POSE_TABLE_RIGHT = (0.6, -0.6)
	POSE_USER = (0.0, 0.0)
	AUTO_TIME = 0.0
	MIN_TIME = 0.5
	SETTLE_TIME = 0.5
	THRESHOLD_JOINT = 0.15

	### Main thread ###

	def __init__(self, use, debug):
		super().__init__(use, debug=debug)
		rospy.loginfo(f"Head control {'enabled' if self.use else 'disabled'}")
		self.current_cmd = None
		self.next_cmd = None
		self.joint_states = None
		self.last_wait_print_time = -math.inf
		self.sub_joint_states = rospy.Subscriber(self.SUB_JOINT_STATES, sensor_msgs.msg.JointState, self.joint_states_callback, queue_size=1) if self.use else None
		self.srv_set_joint = rospy.ServiceProxy(self.SRV_SET_JOINT, open_manipulator_msgs.srv.SetJointPosition, persistent=True) if self.use else None

	### Joint states threads ###

	def joint_states_callback(self, msg: sensor_msgs.msg.JointState):
		self.joint_states = msg  # Atomic write

	### Run thread ###

	def step(self):

		while True:

			now = rospy.Time.now().to_sec()

			queue_empty = False
			while not queue_empty:
				if not self.current_cmd and self.next_cmd:
					self.current_cmd = self.next_cmd
					self.next_cmd = None
				if not self.next_cmd:
					try:
						self.next_cmd: HeadCmd = self.cmd_queue.get_nowait()
					except queue.Empty:
						queue_empty = True
				if self.current_cmd and self.next_cmd:
					if self.current_cmd.preemptable:
						self.current_cmd.future.started.set()
						self.current_cmd.future.finished.set()
						self.current_cmd.future.stayed.set()
						if self.debug:
							print(f">>> Head command {self.current_cmd.id} was preempted")
						self.current_cmd = None
					else:
						break

			if self.current_cmd:

				if self.use:
					joint_states = self.joint_states  # Atomic read
					if joint_states:
						joint_error = max(abs(joint_states.position[0] - self.current_cmd.pitch), abs(joint_states.position[1] - self.current_cmd.yaw))
					else:
						joint_error = math.inf
				else:
					joint_error = 0.0

				if self.current_cmd.start_time == 0:
					if self.current_cmd.move_time <= 0:
						self.current_cmd.move_time = self.MIN_TIME + min(joint_error, 0.5 * math.pi) / 0.7
					self.current_cmd.move_time = max(self.current_cmd.move_time, self.MIN_TIME)
					self.current_cmd.start_time = now
					self.current_cmd.start_success = True
					self.current_cmd.future.started.set()
					if self.debug:
						print(f">>> Head command {self.current_cmd.id} started = {self.current_cmd}")
					if not self.send_request(srv=self.srv_set_joint, request_type='Set head joint pose', response_field='is_planned', request=open_manipulator_msgs.srv.SetJointPositionRequest(
						joint_position=open_manipulator_msgs.msg.JointPosition(position=(self.current_cmd.pitch, self.current_cmd.yaw), max_accelerations_scaling_factor=1, max_velocity_scaling_factor=1),
						path_time=self.current_cmd.move_time,
					)):
						rospy.logwarn("Failed to command head via set joint position service")
						self.current_cmd.start_success = False

				if self.current_cmd.finish_time == 0:
					elapsed_time = now - self.current_cmd.start_time
					if timed_out := (elapsed_time >= self.current_cmd.move_time + self.SETTLE_TIME):
						rospy.logwarn(f"Head command timed out while waiting for joint errors to reduce below {self.THRESHOLD_JOINT:.2g}")
						arrived = True
					elif not self.use or not self.current_cmd.start_success:
						arrived = True
					else:
						if self.debug and now - self.last_wait_print_time >= DEBUG_THROTTLE_TIME:
							print(f">>> Head waiting {elapsed_time:.3f}s for joint error {joint_error:.3g} < {self.THRESHOLD_JOINT:.2g}")
							self.last_wait_print_time = now
						arrived = joint_error < self.THRESHOLD_JOINT
					if (arrived and elapsed_time >= self.current_cmd.move_time) or timed_out:
						self.current_cmd.finish_time = now
						self.current_cmd.future.finished.set()
						if self.debug:
							print(f">>> Head command {self.current_cmd.id} finished in {elapsed_time:.3f}s")

				if self.current_cmd.finish_time != 0:
					elapsed_time = now - self.current_cmd.finish_time
					if elapsed_time >= self.current_cmd.stay_time:
						self.current_cmd.future.stayed.set()
						if self.debug:
							print(f">>> Head command {self.current_cmd.id} stayed for {elapsed_time:.3f}s")
						self.current_cmd = None
						continue

			break

	### Any thread ###

	def is_neutral(self, cmd) -> bool:
		return cmd.pitch == self.POSE_NEUTRAL[0] and cmd.yaw == self.POSE_NEUTRAL[1]

	def set_neutral(self, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None, move_time=AUTO_TIME, preemptable=False) -> ActionFutureBase:
		return self.set_pose(self.POSE_NEUTRAL[0], self.POSE_NEUTRAL[1], stay_time=stay_time, wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout, move_time=move_time, preemptable=preemptable)

	def set_user(self, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None, move_time=AUTO_TIME, preemptable=False):
		return self.set_pose(self.POSE_USER[0], self.POSE_USER[1], stay_time=stay_time, wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout, move_time=move_time, preemptable=preemptable)

	def set_pose(self, pitch, yaw, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None, move_time=AUTO_TIME, preemptable=False):
		future = self.enqueue_command(HeadCmd(pitch=pitch, yaw=yaw, move_time=move_time, stay_time=stay_time, preemptable=preemptable))
		future.wait(wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
		return future

#
# Facial expression
#

# Facial expresssion command class
@dataclasses.dataclass
class FacialExpressionCmd:
	expression: str          # Facial expression to display
	trained: bool            # Whether to use the trained expression variant
	stay_time: float         # Minimum time that the expression should stay displayed before it can be changed again
	display_time: float = 0  # ROS time the command was displayed at (used in the step control loop)
	future: ActionFuture = dataclasses.field(default_factory=ActionFuture)                      # Future that can be used to wait for action events
	id: int = dataclasses.field(default_factory=itertools.count(start=1).__next__, init=False)  # Automatic unique integer ID

	def __repr__(self):
		return f"{self.__class__.__name__}(id={self.id}, expr={self.expression}, trained={self.trained}, stay={self.stay_time:.3g}s)"

# Facial expression class
class FacialExpression(AsyncActionManager):

	EXPRESSION_NEUTRAL = 'neutral'
	EXPRESSION_HAPPY = 'happiness'
	EXPRESSIONS = {EXPRESSION_NEUTRAL: True, EXPRESSION_HAPPY: True, 'sadness': True, 'anger': False, 'surprise': False}
	SRV_FACIAL_EXPRESSION = '/NICOL/face_expression_service'

	### Main thread ###

	def __init__(self, use, debug):
		super().__init__(use, debug=debug)
		rospy.loginfo(f"Facial expressions {'enabled' if self.use else 'disabled'}")
		self.current_cmd: Optional[FacialExpressionCmd] = None
		self.srv_facial_expression = rospy.ServiceProxy(self.SRV_FACIAL_EXPRESSION, nias_msgs.srv.FaceExpression, persistent=True) if self.use else None

	### Run thread ###

	def step(self):

		while True:

			now = rospy.Time.now().to_sec()

			if not self.current_cmd:
				try:
					self.current_cmd: FacialExpressionCmd = self.cmd_queue.get_nowait()
				except queue.Empty:
					break

			if self.current_cmd.display_time == 0:
				self.current_cmd.display_time = now
				self.current_cmd.future.started.set()
				if self.debug:
					print(f">>> Face command {self.current_cmd.id} started = {self.current_cmd}")
				if not self.send_request(srv=self.srv_facial_expression, request_type='Set facial expression', response_field='success', request=nias_msgs.srv.FaceExpressionRequest(face_expression=self.current_cmd.expression, trained=self.current_cmd.trained)):
					rospy.logwarn(f"Failed to set facial expression: {self.current_cmd.expression} (trained={self.current_cmd.trained})")
				self.current_cmd.future.finished.set()
				if self.debug:
					print(f">>> Face command {self.current_cmd.id} finished")

			elapsed_time = now - self.current_cmd.display_time
			if elapsed_time >= self.current_cmd.stay_time:
				self.current_cmd.future.stayed.set()
				if self.debug:
					print(f">>> Face command {self.current_cmd.id} stayed for {elapsed_time:.3f}s")
				self.current_cmd = None
				continue

			break

	### Any thread ###

	def is_neutral(self, cmd) -> bool:
		return cmd.expression == self.EXPRESSION_NEUTRAL

	def set_neutral(self, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None) -> ActionFutureBase:
		return self.set_expression(self.EXPRESSION_NEUTRAL, stay_time=stay_time, wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)

	def set_expression(self, expression, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None):
		trained = self.EXPRESSIONS.get(expression, None)
		if trained is None:
			rospy.logwarn(f"Unrecognised expression: {expression}")
			return None
		future = self.enqueue_command(FacialExpressionCmd(expression=expression, trained=trained, stay_time=stay_time))
		future.wait(wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
		return future

#
# Arm control
#

# Arm command class
@dataclasses.dataclass
class ArmCmd:
	arm_target: Union[tuple[float, ...], geometry_msgs.msg.Pose, None]  # Optional joint/inv arm target pose
	hand_target: Union[tuple[float, ...], None]                         # Optional joint hand target pose
	move_time: float            # Desired move time in seconds (0 = Auto)
	stay_time: float            # Minimum time to stay in the target pose before starting any subsequent arm commands
	preemptable: bool           # Whether any future arm commands immediately overwrite this one, even if this command has not finished yet
	start_time: float = 0       # ROS time the command was started at (used in the step control loop)
	start_success: bool = True  # Whether the start was successful (used in the step control loop, service call and planning was successful, valid if start_time != 0)
	timeout_time: float = 0     # Time after which the command times out if it is still running (used in the step control loop, valid if start_time != 0)
	finish_time: float = 0      # ROS time the command was finished at (used in the step control loop)
	arm_planned: Union[tuple[float, ...], geometry_msgs.msg.Pose, None] = None                  # Arm target that was actually planned (valid if start_time != 0)
	future: ActionFuture = dataclasses.field(default_factory=ActionFuture)                      # Future that can be used to wait for action events
	id: int = dataclasses.field(default_factory=itertools.count(start=1).__next__, init=False)  # Automatic unique integer ID

	def __repr__(self):
		if isinstance(self.arm_target, geometry_msgs.msg.Pose):
			arm_target = f"({self.arm_target.position.x:.3g}, {self.arm_target.position.y:.3g}, {self.arm_target.position.z:.3g})+({self.arm_target.orientation.w:.3g}, {self.arm_target.orientation.x:.3g}, {self.arm_target.orientation.y:.3g}, {self.arm_target.orientation.z:.3g})"
		else:
			arm_target = self.arm_target and format_tuple(self.arm_target, '.3g')
		hand_target = self.hand_target and format_tuple(self.hand_target, '.3g')
		return f"{self.__class__.__name__}(id={self.id}, arm={arm_target}, hand={hand_target}, move={self.move_time:.3g}s, stay={self.stay_time:.3g}s, preemptable={self.preemptable})"

# Arm control class
class ArmControl(AsyncActionManager):

	ARM_CONFIG = ((False, 1, 'L', '/left', 'l_arm', 'l_laser'), (True, -1, 'R', '/right', 'r_arm', 'r_laser'))
	URI_SET_ARM_JOINT = '/open_manipulator_p/goal_joint_space_path'             # Hand joint order: (index finger, ring pinky fingers, middle finger, thumb direction, thumb curl)
	URI_SET_HAND_JOINT = '/open_manipulator_p/goal_joint_space_path_hand_only'  # Hand joint order: (thumb direction, thumb curl, index finger, middle finger, ring pinky fingers)
	URI_SET_ARM_INV = '/open_manipulator_p/goal_task_space_path'
	URI_GET_ARM_INV = '/open_manipulator_p/moveit/get_kinematics_pose'
	URI_GET_ARM_JOINT = '/open_manipulator_p/joint_states'
	ARM_TARGET_HOME = (-1.35, -0.8, 0.4, 0.0, 0.4, 0.0, 0.3, 0.0)
	HAND_TARGET_OPEN = (-1.5, 0.3, -2.8, -2.8, -2.8)  # Note: These are in hand joint convention (thumb first)
	HAND_TARGET_CLOSED = (-1.5, 0.3, 1.0, 1.0, 1.0)
	HAND_TARGET_HOME = HAND_TARGET_CLOSED
	AUTO_TIME = 0.0
	MIN_ARM_TIME = 0.5
	MIN_ARM_TIME_SIM = 4.0
	HAND_ONLY_TIME = 1.2
	INV_ARM_TIMEOUT = 16.0
	RESET_FIRST_STAY_TIME = 0.1
	SETTLE_TIME = 0.5
	SETTLE_SCALER = 1.25
	JOINT_ACC_TIME = 1.2
	JOINT_VEL_LIMIT = 0.105
	THRESHOLD_JOINT = 0.03
	THRESHOLD_POS = 0.01
	THRESHOLD_ROT = 0.07

	### Main thread ###

	def __init__(self, use, debug, is_sim, is_right):

		super().__init__(use, debug=debug)
		self.is_sim = is_sim
		self.is_right, self.arm_sign, self.letter, self.prefix, self.planning_group, self.end_effector_name = self.ARM_CONFIG[is_right]
		self.name = self.letter + 'Arm'
		assert self.is_right == is_right
		rospy.loginfo(f"{self.name} control {'enabled' if self.use else 'disabled'}")

		self.arm_target_home = self.resolve_joint_arm_target(self.ARM_TARGET_HOME)
		self.hand_target_home = self.HAND_TARGET_HOME
		self.min_arm_time = self.MIN_ARM_TIME_SIM if self.is_sim else self.MIN_ARM_TIME

		self.current_cmd = None
		self.next_cmd = None
		self.joint_states = None
		self.last_wait_print_time = -math.inf
		self.last_srv_get_arm_inv_fail_time = -math.inf

		self.srv_set_arm_joint = rospy.ServiceProxy(self.prefix + self.URI_SET_ARM_JOINT, open_manipulator_msgs.srv.SetJointPosition, persistent=True) if self.use else None
		self.srv_set_hand_joint = rospy.ServiceProxy(self.prefix + self.URI_SET_HAND_JOINT, open_manipulator_msgs.srv.SetJointPosition, persistent=True) if self.use else None
		self.srv_set_arm_inv = rospy.ServiceProxy(self.prefix + self.URI_SET_ARM_INV, open_manipulator_msgs.srv.SetKinematicsPose, persistent=True) if self.use else None
		self.srv_get_arm_inv = rospy.ServiceProxy(self.prefix + self.URI_GET_ARM_INV, open_manipulator_msgs.srv.GetKinematicsPose, persistent=True) if self.use else None
		self.sub_get_arm_joint = rospy.Subscriber(self.prefix + self.URI_GET_ARM_JOINT, sensor_msgs.msg.JointState, callback=self.joint_states_callback, queue_size=1) if self.use else None

	### Joint states threads ###

	def joint_states_callback(self, msg: sensor_msgs.msg.JointState):
		self.joint_states = msg  # Atomic write

	### Run thread ###

	def step(self):

		while True:

			now = rospy.Time.now().to_sec()

			queue_empty = False
			while not queue_empty:
				if not self.current_cmd and self.next_cmd:
					self.current_cmd = self.next_cmd
					self.next_cmd = None
				if not self.next_cmd:
					try:
						next_cmd: ArmCmd = self.cmd_queue.get_nowait()
						if next_cmd:
							if next_cmd.arm_target is None and next_cmd.hand_target is None:
								rospy.logwarn(f"{self.name} received an all-None command")
							elif isinstance(next_cmd.arm_target, geometry_msgs.msg.Pose) and next_cmd.hand_target:
								rospy.logwarn(f"{self.name} received an inverse arm target with a joint hand target")
							elif isinstance(next_cmd.arm_target, tuple) and len(next_cmd.arm_target) != 8:
								rospy.logwarn(f"{self.name} received a joint arm target with {len(next_cmd.arm_target)} != 8 values")
							elif next_cmd.hand_target and len(next_cmd.hand_target) != 5:
								rospy.logwarn(f"{self.name} received a joint hand target with {len(next_cmd.hand_target)} != 5 values")
							else:
								self.next_cmd = next_cmd
					except queue.Empty:
						queue_empty = True
				if self.current_cmd and self.next_cmd:
					if self.current_cmd.preemptable:
						self.current_cmd.future.started.set()
						self.current_cmd.future.finished.set()
						self.current_cmd.future.stayed.set()
						if self.debug:
							print(f">>> {self.name} command {self.current_cmd.id} was preempted")
						self.current_cmd = None
					else:
						break

			if self.current_cmd:

				if self.current_cmd.start_time == 0:
					if self.current_cmd.move_time <= 0:
						if isinstance(self.current_cmd.arm_target, tuple):
							joint_states = self.joint_states  # Atomic read
							if joint_states:
								joint_move_max = max(abs(target - state) for target, state in zip(self.current_cmd.arm_target, joint_states.position))
							else:
								joint_move_max = math.pi / 4
							self.current_cmd.move_time = self.JOINT_ACC_TIME + joint_move_max / self.JOINT_VEL_LIMIT
						elif isinstance(self.current_cmd.arm_target, geometry_msgs.msg.Pose):
							self.current_cmd.move_time = self.min_arm_time  # This time is ignored anyway by the service server...
						else:
							self.current_cmd.move_time = self.HAND_ONLY_TIME
					if self.current_cmd.arm_target:
						self.current_cmd.move_time = max(self.current_cmd.move_time, self.min_arm_time)
					if self.current_cmd.hand_target:
						self.current_cmd.move_time = max(self.current_cmd.move_time, self.HAND_ONLY_TIME)
					if isinstance(self.current_cmd.arm_target, tuple):
						self.current_cmd.timeout_time = self.SETTLE_SCALER * self.current_cmd.move_time + self.SETTLE_TIME
					elif isinstance(self.current_cmd.arm_target, geometry_msgs.msg.Pose):
						self.current_cmd.timeout_time = max(self.current_cmd.move_time + self.SETTLE_TIME, self.INV_ARM_TIMEOUT)
					else:
						self.current_cmd.timeout_time = self.current_cmd.move_time + self.SETTLE_TIME
					self.current_cmd.arm_planned = self.current_cmd.arm_target
					self.current_cmd.start_time = now
					self.current_cmd.start_success = True
					self.current_cmd.future.started.set()
					if self.debug:
						print(f">>> {self.name} command {self.current_cmd.id} started = {self.current_cmd}")
					if isinstance(self.current_cmd.arm_target, tuple):
						joint_target = self.current_cmd.arm_target
						if self.current_cmd.hand_target:
							joint_target += (self.current_cmd.hand_target[2], self.current_cmd.hand_target[4], self.current_cmd.hand_target[3], self.current_cmd.hand_target[0], self.current_cmd.hand_target[1])
						if not self.send_request(srv=self.srv_set_arm_joint, request_type='Set arm joint pose', response_field='is_planned', request=open_manipulator_msgs.srv.SetJointPositionRequest(
							planning_group=self.planning_group,
							joint_position=open_manipulator_msgs.msg.JointPosition(position=joint_target, max_accelerations_scaling_factor=1, max_velocity_scaling_factor=1),
							path_time=self.current_cmd.move_time,
						)):
							rospy.logwarn(f"{self.name} failed to set arm joint pose via service call")
							self.current_cmd.start_success = False
					elif isinstance(self.current_cmd.arm_target, geometry_msgs.msg.Pose):
						if response := self.send_request(srv=self.srv_set_arm_inv, request_type='Set arm inv pose', response_field='is_planned', request=open_manipulator_msgs.srv.SetKinematicsPoseRequest(
							planning_group=self.planning_group,
							end_effector_name=self.end_effector_name,
							kinematics_pose=open_manipulator_msgs.msg.KinematicsPose(pose=self.current_cmd.arm_target, max_accelerations_scaling_factor=1, max_velocity_scaling_factor=1, tolerance=0),
							path_time=self.current_cmd.move_time,
						), nonuse_response=open_manipulator_msgs.srv.SetKinematicsPoseResponse(is_planned=True, pose=self.current_cmd.arm_target)):
							P = response.pose
							T = self.current_cmd.arm_target
							pos_error = math.sqrt((P.position.x - T.position.x) ** 2 + (P.position.y - T.position.y) ** 2 + (P.position.z - T.position.z) ** 2)
							rot_error = 2 * math.acos(min(1.0, abs(P.orientation.w * T.orientation.w + P.orientation.x * T.orientation.x + P.orientation.y * T.orientation.y + P.orientation.z * T.orientation.z)))
							if self.debug:
								print(f">>> {self.name} inverse planning error is pos err {pos_error:.3f} and rot err {rot_error:.3f}")
							if pos_error > self.THRESHOLD_POS or rot_error > self.THRESHOLD_ROT:
								rospy.logwarn(f"{self.name} inverse planning error is significant, with pos err {pos_error:.3f} and rot err {rot_error:.3f}")
							self.current_cmd.arm_planned = P
						else:
							rospy.logwarn(f"{self.name} failed to set arm inv pose via service call")
							self.current_cmd.start_success = False
					elif self.current_cmd.hand_target:  # Note: This is the correct way to send hand joint commands to the real robot, but it does not work properly in Gazebo
						if not self.send_request(srv=self.srv_set_hand_joint, request_type='Set hand joint pose', response_field='is_planned', request=open_manipulator_msgs.srv.SetJointPositionRequest(
							planning_group=self.planning_group,
							joint_position=open_manipulator_msgs.msg.JointPosition(position=self.current_cmd.hand_target, max_accelerations_scaling_factor=1, max_velocity_scaling_factor=1),
							path_time=self.current_cmd.move_time,
						)):
							rospy.logwarn(f"{self.name} failed to set hand joint pose via service call")
							self.current_cmd.start_success = False
					else:
						rospy.logwarn(f"{self.name} did not have any service call to call")
						self.current_cmd.start_success = False

				if self.current_cmd.finish_time == 0:
					elapsed_time = now - self.current_cmd.start_time
					if timed_out := (elapsed_time >= self.current_cmd.timeout_time):
						rospy.logwarn(f"{self.name} command timed out while waiting for joint errors to reduce below {self.THRESHOLD_JOINT:.2g}")
						arrived = True
					elif not self.use or not self.current_cmd.start_success:
						arrived = True
					elif isinstance(self.current_cmd.arm_planned, tuple):
						joint_states = self.joint_states  # Atomic read
						if joint_states:
							joint_errors = tuple(state - target for state, target in zip(joint_states.position, self.current_cmd.arm_planned))
							if self.debug and now - self.last_wait_print_time >= DEBUG_THROTTLE_TIME:
								print(f">>> {self.name} waiting {elapsed_time:.3f}s for all joints <{self.THRESHOLD_JOINT:.2g}: {format_tuple(joint_errors, ' .3f')}")
								self.last_wait_print_time = now
							arrived = all(abs(joint_error) < self.THRESHOLD_JOINT for joint_error in joint_errors)
						else:
							arrived = False
					elif isinstance(self.current_cmd.arm_planned, geometry_msgs.msg.Pose):
						start = rospy.Time.now()
						response = self.srv_get_arm_inv(open_manipulator_msgs.srv.GetKinematicsPoseRequest(planning_group=self.planning_group, end_effector_name=self.end_effector_name))
						duration = (rospy.Time.now() - start).to_sec()
						if response:
							arm_inv_state = response.kinematics_pose.pose
							pos_error = math.sqrt((arm_inv_state.position.x - self.current_cmd.arm_planned.position.x) ** 2 + (arm_inv_state.position.y - self.current_cmd.arm_planned.position.y) ** 2 + (arm_inv_state.position.z - self.current_cmd.arm_planned.position.z) ** 2)
							rot_error = 2 * math.acos(min(1.0, abs(arm_inv_state.orientation.w * self.current_cmd.arm_planned.orientation.w + arm_inv_state.orientation.x * self.current_cmd.arm_planned.orientation.x + arm_inv_state.orientation.y * self.current_cmd.arm_planned.orientation.y + arm_inv_state.orientation.z * self.current_cmd.arm_planned.orientation.z)))
							if self.debug and now - self.last_wait_print_time >= DEBUG_THROTTLE_TIME:
								print(f">>> {self.name} waiting {elapsed_time:.3f}s for pos err {pos_error:.3f} < {self.THRESHOLD_POS:.2g} and rot err {rot_error:.3f} < {self.THRESHOLD_ROT:.2g} (service took {duration:.3f}s)")
								self.last_wait_print_time = now
							arrived = pos_error < self.THRESHOLD_POS and rot_error < self.THRESHOLD_ROT
						else:
							if now - self.last_srv_get_arm_inv_fail_time >= DEBUG_THROTTLE_TIME:
								rospy.logwarn(f"{self.name} failed to get arm inv pose via service call")
								self.last_srv_get_arm_inv_fail_time = now
							arrived = False
					else:
						arrived = True
					if (arrived and elapsed_time >= self.current_cmd.move_time) or timed_out:
						self.current_cmd.finish_time = now
						self.current_cmd.future.finished.set()
						if self.debug:
							print(f">>> {self.name} command {self.current_cmd.id} finished in {elapsed_time:.3f}s")

				if self.current_cmd.finish_time != 0:
					elapsed_time = now - self.current_cmd.finish_time
					if elapsed_time >= self.current_cmd.stay_time:
						self.current_cmd.future.stayed.set()
						if self.debug:
							print(f">>> {self.name} command {self.current_cmd.id} stayed for {elapsed_time:.3f}s")
						self.current_cmd = None
						continue

			break

	### Any thread ###

	def resolve_joint_arm_target(self, joint_arm_target: tuple[float, ...]):
		return tuple((-pos if i in (1, 4, 6, 7) else pos) for i, pos in enumerate(joint_arm_target, 1)) if self.is_right else joint_arm_target

	def is_neutral(self, cmd) -> bool:
		return cmd.arm_target == self.arm_target_home and cmd.hand_target == self.hand_target_home

	def set_neutral(self, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None, move_time=AUTO_TIME, preemptable=False) -> ActionFutureBase:
		future = MultiActionFuture(futures=(
			self.set_pose(hand_target=self.hand_target_home, reset_first=False, auto_resolve=False, stay_time=0, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None, move_time=self.AUTO_TIME, preemptable=False),
			self.set_pose(arm_target=self.arm_target_home, hand_target=self.hand_target_home, reset_first=False, auto_resolve=False, stay_time=stay_time, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None, move_time=move_time, preemptable=preemptable),
		))
		future.wait(wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
		return future

	def open_hand(self, reset_first=False, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None, move_time=AUTO_TIME, preemptable=False):
		return self.set_pose(arm_target=None, hand_target=self.HAND_TARGET_OPEN, reset_first=reset_first, auto_resolve=False, stay_time=stay_time, wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout, move_time=move_time, preemptable=preemptable)

	def close_hand(self, reset_first=False, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None, move_time=AUTO_TIME, preemptable=False):
		return self.set_pose(arm_target=None, hand_target=self.HAND_TARGET_CLOSED, reset_first=reset_first, auto_resolve=False, stay_time=stay_time, wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout, move_time=move_time, preemptable=preemptable)

	def set_pose(self, arm_target=None, hand_target=None, reset_first=False, auto_resolve=True, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None, move_time=AUTO_TIME, preemptable=False):
		if arm_target is None and hand_target is None:
			return COMPLETED_ACTION_FUTURE
		if reset_first:
			self.reset_state(stay_time=self.RESET_FIRST_STAY_TIME, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None)
		if auto_resolve and isinstance(arm_target, tuple):
			arm_target = self.resolve_joint_arm_target(arm_target)
		future = self.enqueue_command(ArmCmd(arm_target=arm_target, hand_target=hand_target, move_time=move_time, stay_time=stay_time, preemptable=preemptable))
		future.wait(wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
		return future

#
# Chat demo
#

# Chat status event enumeration
class StatusEvent(Enum):
	Event = nicol_demos.msg.ChatStatus.EVENT
	Started = nicol_demos.msg.ChatStatus.STARTED
	Finished = nicol_demos.msg.ChatStatus.FINISHED

# Dialogue speaker enumeration
class DialogueSpeaker(Enum):
	NewChat = nicol_demos.msg.Dialogue.NEW_CHAT
	User = nicol_demos.msg.Dialogue.USER
	Robot = nicol_demos.msg.Dialogue.ROBOT

# Response type enumeration
# noinspection PyArgumentList
class ResponseType(Enum):
	Say = auto()
	Action = auto()

# Agenda item class
@dataclasses.dataclass(frozen=True)
class AgendaItem:
	type: ResponseType
	text: str
	name: Optional[str] = None
	func: Optional[Callable] = None

# Chat demo class
class ChatDemo(demo_base.DemoBase):

	ACTIONS = {
		'look': (('object',), "Look at {object}", "Given a string of an object name, move your head to look at that object. In any situation, the object name can also be \"user\" in order to look at the user, or it can be \"table\" in order to look at the table."),
		'point': (('object',), "Point at {object}", "Given a string of an object name, use your arms to point to that object on the table. In any situation, the object name can also be \"user\" in order to point at the user."),
		'give': (('object',), "Give {object} to user", "Given a string of an object name, use your arms to grasp that object on the table and give it to the user. You can hand objects to the user with this function."),
		'express': (('emotion',), "Express {emotion}", f"Given a string emotion name, change your facial expression to match that emotion. The list of available emotions is [{', '.join(FacialExpression.EXPRESSIONS)}]."),
	}
	PUB_CHAT_STATUS = '~chat_status'
	PUB_DIALOGUE = '~dialogue'
	EMOTION_STAY_TIME = 2.0
	POSTINIT_OBJECT_DELAY_TIME = 1.5
	HISTFILE = '.chat_demo_history'
	TABLE_DIM = (1.0, 1.0, 0.8)  # Table dimensions (Tx, Ty, Tz) such that table is [0,Tx] x [-Ty,Ty] x {Tz}
	POINT_ARM_TARGET_USER = (-1.45, 0.50, -0.28, 1.92, -0.33, -1.00, -0.19, -0.21)
	POINT_ARM_TARGET_TABLE = (-1.20, -0.94, 0.26, 0.28, 0.63, 0.10, -0.21, 0.45)
	POINT_HAND_TARGET = (-1.5, 0.3, -3.0, 1.0, 1.0)
	POINT_CONTROL_POINT = (0, 0.45, 1.20)  # Virtual point that primarily determines hand orientation, but implicitly also position, in particular height
	POINT_THETA_ABSMAX = 1.0      # Maximum allowed theta deviation in order to proceed with point action
	POINT_GAMMA_MAX = 1.55        # Maximum allowed alpha in order to proceed with point action
	POINT_REACH_MAX = 0.78        # Maximum allowed reach for generated pointing poses
	POINT_TABLE_CLEARANCE = 0.11  # Clearance above table level that should be maintained in the pointing position
	POINT_FINGER_ANGLE = 0.60     # Angle below the laser frame that the finger actually points to, with assumed origin at the laser frame
	POINT_NOMINAL_DIST = 0.25     # Nominal distance from object centre to pointing laser frame
	POINT_DIM_FINGER = 0.095      # Assumed finger length (parallel to hand plane) used to determine suitable hand position height
	POINT_DIM_HEIGHT = 0.045      # Assumed fist thickness (perpendicular to hand plane) used to determine suitable hand position height
	POINT_CHECK_TOL = 0.001       # Tolerance in metres regarding the lambda/psi check (does not error for ever so slight shortening on the scale of floating point errors)
	POINT_OFFSET_L = (0.015, 0.05, 0.0, 0.0)  # Radial along alpha-line (negative means closer to control point and further away from object), tangential (positive means the tangential more towards outside left/right of table), ...
	POINT_OFFSET_R = (0.015, 0.06, 0.0, 0.0)  # ..., X (positive means more towards front of table), Y (positive means more towards outside left/right of table)
	GIVE_HAND_TARGET = (-1.5, 0.3, -1.5, -1.5, -1.5)
	GIVE_CONTROL_POINT = (0, 0.45, 1.20)  # Virtual point that primarily determines hand orientation and stroke direction, but also affects hand height
	GIVE_THETA_ABSMAX = 1.0  # Maximum allowed theta deviation in order to proceed with give action
	GIVE_ALPHA_MAX = 1.25    # Maximum allowed alpha in order to proceed with give action
	GIVE_DIM_FINGER = 0.095  # Assumed finger length (parallel to line from control point to hand position) used to determine suitable hand position height
	GIVE_DIM_HEIGHT = 0.07   # Finger tip height (perpendicular to line from control point to hand/finger tip position) used to determine suitable hand position height
	GIVE_DESCENT = 0.08      # Distance to vertically descend prior to the stroke
	GIVE_STROKE = 0.08       # Distance to move radially forward while opening the hand to signal a give
	GIVE_CLEARANCE = 0.01    # Distance to move vertically downward while performing forward-radial stroke
	GIVE_OFFSET_L = (0.0, 0.0, 0.0, 0.0)  # Radial (negative means closer to control point and further away from object), tangential (positive means the tangential more towards outside left/right of table), ...
	GIVE_OFFSET_R = (0.0, 0.0, 0.0, 0.0)  # ..., X (positive means more towards front of table), Y (positive means more towards outside left/right of table)

	### Main thread ###

	def __init__(self, use_objects, use_human, use_asr, fixed_objects, fixed_objects_pose, asr_detect, asr_duration, use_speech, use_head, use_face, use_arms, use_model, model_response, query_facts, model_verbose, clear_history, debug_asr, debug_head, debug_face, debug_arms, is_sim):

		self.use_objects = use_objects
		self.use_human = use_human
		self.use_asr = use_asr
		self.fixed_objects = fixed_objects
		self.fixed_objects_pose = fixed_objects_pose
		self.asr_detect = asr_detect
		self.asr_duration = asr_duration
		self.use_speech = use_speech
		self.use_head = use_head
		self.use_face = use_face
		self.use_arms = use_arms
		self.use_model = use_model
		self.model_response = model_response
		self.query_facts = query_facts
		self.model_verbose = model_verbose
		self.clear_history = clear_history
		self.debug_asr = debug_asr
		self.debug_head = debug_head
		self.debug_face = debug_face
		self.debug_arms = debug_arms
		self.is_sim = is_sim
		if self.is_sim:
			rospy.logwarn("Simulation mode is active")
		else:
			rospy.loginfo("Real robot mode is active")

		super().__init__()

		self.pub_chat_status = None
		self.pub_dialogue = None

		self.chat_manager: Optional[ChatManager] = None
		self.object_detector: Optional[ObjectDetector] = None
		self.asr_manager: Optional[ASRManager] = None
		self.speech_manager: Optional[SpeechManager] = None
		self.head_control: Optional[HeadControl] = None
		self.facial_expression: Optional[FacialExpression] = None
		self.arm_control_l: Optional[ArmControl] = None
		self.arm_control_r: Optional[ArmControl] = None
		self.arm_controls = (self.arm_control_l, self.arm_control_r)

		self.current_objects = {}

		self.chat_demo_histfile = os.path.join(rospkg.get_ros_home(), self.HISTFILE)
		if self.clear_history:
			with open(self.chat_demo_histfile, 'w'):
				self.orig_histfile_len = 0
		else:
			try:
				readline.read_history_file(self.chat_demo_histfile)
				self.orig_histfile_len = readline.get_current_history_length()
			except FileNotFoundError:
				with open(self.chat_demo_histfile, 'w'):
					self.orig_histfile_len = 0

	def prepare_run(self):

		self.pub_chat_status = ROSPublisher(self.PUB_CHAT_STATUS, nicol_demos.msg.ChatStatus, queue_size=100)
		self.pub_dialogue = ROSPublisher(self.PUB_DIALOGUE, nicol_demos.msg.Dialogue, queue_size=100)
		self.pub_chat_status.wait_connected()
		self.pub_dialogue.wait_connected()
		rospy.on_shutdown(self.on_exit)

		with self.publish_status('prepare'):
			self.chat_manager = ChatManager(actions=self.ACTIONS, use_model=self.use_model, model_response=self.model_response, query_facts=self.query_facts, model_verbose=self.model_verbose)
			self.object_detector = ObjectDetector(use=self.use_objects, fixed_objects=self.fixed_objects, fixed_objects_pose=self.fixed_objects_pose)
			self.asr_manager = ASRManager(use=self.use_asr, detect=self.asr_detect, max_duration=self.asr_duration, verbose=self.debug_asr)
			self.speech_manager = SpeechManager(use=self.use_speech)
			self.head_control = HeadControl(use=self.use_head, debug=self.debug_head)
			self.facial_expression = FacialExpression(use=self.use_face, debug=self.debug_face)
			self.arm_control_l = ArmControl(use=self.use_arms, debug=self.debug_arms, is_sim=self.is_sim, is_right=False)
			self.arm_control_r = ArmControl(use=self.use_arms, debug=self.debug_arms, is_sim=self.is_sim, is_right=True)
			self.arm_controls = (self.arm_control_l, self.arm_control_r)
			super().prepare_run()

	def on_exit(self):
		self.publish_event('exit')
		final_histfile_len = readline.get_current_history_length()
		readline.set_history_length(2000)
		readline.append_history_file(final_histfile_len - self.orig_histfile_len, self.chat_demo_histfile)

	def main(self):

		with self.publish_status('init'):
			rospy.loginfo("Initialising robot to neutral state...")
			reset_future = MultiActionFuture(futures=(
				self.arm_control_l.open_hand(stay_time=0, wait_finished=False),
				self.arm_control_r.open_hand(stay_time=0, wait_finished=False),
				self.reset_state(stay_time=0, wait_finished=False),
			))
			self.chat_manager.init_chat()
			print("\x1b[2K\rEXECUTE: Initialising robot to neutral state... ", end='', flush=True)
			reset_future.wait_finished()
			print("\x1b[2K\r", end='', flush=True)

		self.publish_dialogue(DialogueSpeaker.NewChat, None)
		print()
		if not self.query_facts:
			print("\x1b[2K\rOBJECTS: Giving object detections time to stabilise... ", end='', flush=True)
			time.sleep(self.POSTINIT_OBJECT_DELAY_TIME)
			print("\x1b[2K\r", end='', flush=True)
		self.update_current_objects(newline=False, status=True)

		while True:

			with self.publish_status('user_text'):
				user_text = self.asr_manager.get_user_text()
			user_text_clean = user_text.lower().rstrip('.!')
			if user_text_clean in ('quit', 'exit'):
				break
			elif user_text_clean in ('clear', 'reset'):
				self.publish_event('reset_chat')
				self.publish_dialogue(DialogueSpeaker.NewChat, None)
				self.chat_manager.reset_chat()
				continue
			self.publish_dialogue(DialogueSpeaker.User, user_text)
			print()

			self.update_current_objects(newline=True, status=False)

			with self.publish_status('chat_model'):
				nicol_msg = self.chat_manager.generate_response_user(user_text, progress='SAY')
			nicol_text = nicol_msg.content

			agenda = []
			speech_parts = []
			nicol_text = re.sub(r'<([^\W_]+)[_ ]([\w ]+)>', lambda m: f"<{m.group(1)}({m.group(2).replace('_', ' ')})>", nicol_text)
			for i, text_part in enumerate(re.split(r'(<\w+\([\w ]+\)>)', nicol_text)):
				if i % 2 == 0:
					for j, text_subpart in enumerate(re.split(r'((?<!\w)\([^()]*(?:\([^()]*\))?[^()]*\)(?!>)|\*[^*]+\*)', text_part)):
						is_thought = (j % 2 == 1)
						if is_thought:
							text_subpart = text_subpart.strip('()*')
						text_subpart = re.sub(r'<[^<>\n]*>', r'', text_subpart)
						for k, text_fragment in enumerate(clean_text(fragment) for fragment in (text_subpart + '\n').splitlines()):
							if k > 0 and speech_parts:
								agenda.append(AgendaItem(type=ResponseType.Say, text=capitalize(' '.join(speech_parts))))
								speech_parts.clear()
							if text_fragment:
								print(f"{'THINK' if is_thought else 'SAY'}: {capitalize(text_fragment)}")
								if not is_thought:
									speech_parts.append(text_fragment)
				else:
					match = re.fullmatch(r'<(\w+)\(([\w ]+)\)>', text_part)
					if match:
						action = match.group(1)
						action_args = tuple(arg.strip() for arg in match.group(2).split(','))
						action_info = self.ACTIONS.get(action, None)
						if action_info:
							action_arg_names, action_summary, action_desc = action_info
							if len(action_args) == len(action_arg_names):
								action_kwargs = dict(zip(action_arg_names, action_args))
								action_summary_str = action_summary.format(**action_kwargs)
								if action_args == action_arg_names:
									text_fragment = f'invoke "{action_summary_str}"'
									speech_parts.append(text_fragment)
									print(f"SAY: {capitalize(text_fragment)}")
								else:
									print(f"ACTION: {action_summary_str}")
									action_method = getattr(self, f"action_{action}", None)
									if not action_method:
										raise NotImplementedError(f"Backend for action '{action}' has not been implemented yet")
									if speech_parts:
										agenda.append(AgendaItem(type=ResponseType.Say, text=capitalize(' '.join(speech_parts))))
										speech_parts.clear()
									agenda.append(AgendaItem(type=ResponseType.Action, text=action_summary_str, name=action, func=functools.partial(action_method, **action_kwargs)))
			if speech_parts:
				agenda.append(AgendaItem(type=ResponseType.Say, text=capitalize(' '.join(speech_parts))))

			print()
			self.chat_manager.show_model_stats()

			have_spoken = False
			texts_to_speak = []
			speak_texts = [item.text for item in agenda if item.type == ResponseType.Say]
			for item in itertools.chain(agenda, (AgendaItem(type=ResponseType.Action, text='', name=None, func=None),)):
				if item.type == ResponseType.Say:
					texts_to_speak.append(item.text)
				elif item.type == ResponseType.Action:
					if texts_to_speak:
						num_speak = len(texts_to_speak)
						print(f"\x1b[2K\rEXECUTE: Waiting for {num_speak} speech items... ", end='', flush=True)
						self.publish_dialogue(DialogueSpeaker.Robot, texts_to_speak)
						with self.publish_status('say'):
							if have_spoken:
								self.speech_manager.speak_cached(num_speak)
							else:
								self.speech_manager.speak(speak_texts, num_speak)
						print("\x1b[2K\r", end='', flush=True)
						texts_to_speak.clear()
						have_spoken = True
					if callable(item.func):
						print(f"\x1b[2K\rEXECUTE: {item.text}... ", end='', flush=True)
						with self.publish_status(f'action_{item.name}'):
							item.func()
						print("\x1b[2K\r", end='', flush=True)
				else:
					raise ValueError(f"Unexpected response type: {item.type}")
			self.speech_manager.clear()

			print("\x1b[2K\rEXECUTE: Returning robot to neutral state... ", end='', flush=True)
			with self.publish_status('reset_neutral'):
				self.reset_state(stay_time=0, wait_finished=True)
			print("\x1b[2K\r", end='', flush=True)

	def update_current_objects(self, newline=True, status=False):
		print("\x1b[2K\rOBJECTS: Waiting for object detections... ", end='', flush=True)
		with self.publish_status('objects'):
			self.current_objects = self.object_detector.get_current_objects()
		print("\x1b[2K\r", end='', flush=True)
		if self.chat_manager.update_objects(objects=self.current_objects):
			if newline:
				print()
			if status:
				self.chat_manager.generate_status_update()

	@contextlib.contextmanager
	def publish_status(self, item):
		self.publish_status_msg(item, StatusEvent.Started)
		yield
		self.publish_status_msg(item, StatusEvent.Finished)

	def publish_event(self, item):
		self.publish_status_msg(item, StatusEvent.Event)

	def publish_status_msg(self, item: str, status: StatusEvent):
		msg = nicol_demos.msg.ChatStatus(header=std_msgs.msg.Header(stamp=rospy.Time.now()), item=item, status=status.value)
		self.pub_chat_status.publish(msg)

	def publish_dialogue(self, speaker: DialogueSpeaker, texts: Union[None, str, Iterable[str]]):
		texts = [] if texts is None else [texts] if isinstance(texts, str) else list(texts)
		msg = nicol_demos.msg.Dialogue(header=std_msgs.msg.Header(stamp=rospy.Time.now()), speaker=speaker.value, texts=texts)
		self.pub_dialogue.publish(msg)

	def reset_state(self, stay_time=DEFAULT_STAY_TIME, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None):
		future = MultiActionFuture(futures=(
			self.head_control.reset_state(stay_time=stay_time, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None),
			self.facial_expression.reset_state(stay_time=stay_time, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None),
			self.arm_control_l.reset_state(stay_time=stay_time, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None),
			self.arm_control_r.reset_state(stay_time=stay_time, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None),
		))
		future.wait(wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
		return future

	# noinspection PyShadowingBuiltins
	def action_look(self, object):
		self.perform_look(object)

	# noinspection PyShadowingBuiltins
	def perform_look(self, object, wait_started=False, wait_finished=True, wait_stayed=False, timeout=None):
		if object in ('user', 'human', 'person'):
			return self.head_control.set_user(wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
		elif object == 'table':
			self.head_control.set_pose(*HeadControl.POSE_TABLE_LEFT, stay_time=0.2, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None)
			self.head_control.set_pose(*HeadControl.POSE_TABLE_RIGHT, stay_time=0.2, wait_started=False, wait_finished=False, wait_stayed=False, timeout=None)
			return self.head_control.set_neutral(wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
		elif object in ('neutral', 'down', 'object'):
			return self.head_control.set_neutral(wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
		else:
			detected_object: nicol_demos.msg.DetectedObject = self.current_objects.get(object, None)
			if detected_object:
				return self.head_control.set_pose(detected_object.pitch, detected_object.yaw, wait_started=wait_started, wait_finished=wait_finished, wait_stayed=wait_stayed, timeout=timeout)
			else:
				rospy.logwarn(f"Cannot look at unavailable/invalid object '{object}'")
				return COMPLETED_ACTION_FUTURE

	# noinspection PyShadowingBuiltins
	def action_point(self, object):
		if object in ('user', 'human', 'person'):
			head_future = self.perform_look(object, wait_finished=False)
			arm_control = random.choice(self.arm_controls)
			arm_control.set_pose(arm_target=self.POINT_ARM_TARGET_USER, hand_target=self.POINT_HAND_TARGET, reset_first=True)
		elif object == 'table':
			head_future = self.perform_look('neutral', wait_finished=False)
			futures = {}
			for arm_control in self.arm_controls:
				futures[arm_control] = arm_control.set_pose(arm_target=self.POINT_ARM_TARGET_TABLE, hand_target=self.POINT_HAND_TARGET, reset_first=True, wait_finished=False)
			for future in futures.values():
				future.wait(wait_finished=True)
		else:
			detected_object: nicol_demos.msg.DetectedObject = self.current_objects.get(object, None)
			if detected_object:
				head_future = self.perform_look(object, wait_finished=False)
				Bx = detected_object.centre.x
				By = detected_object.centre.y
				Bz = detected_object.centre.z
				Tz = self.TABLE_DIM[2]
				if not (0 <= Bx <= self.TABLE_DIM[0] and abs(By) <= self.TABLE_DIM[1] and 0 <= Bz - Tz <= 0.5 * self.POINT_TABLE_CLEARANCE):
					rospy.logwarn(f"Cannot point at object '{object}' at position ({Bx:.3f}, {By:.3f}, {Bz:.3f}) due to table dimensions ({self.TABLE_DIM[0]:.3f}, {self.TABLE_DIM[1]:.3f}, {Tz:.3f}) and clearance {self.POINT_TABLE_CLEARANCE:.3f}")
				else:
					arm_control = self.get_arm_control(ycoord=By)
					Cx, Cy, Cz = self.POINT_CONTROL_POINT
					Cy *= arm_control.arm_sign
					CBx = Bx - Cx
					CBy = By - Cy
					CBz = Bz - Cz
					CBrsq = CBx ** 2 + CBy ** 2
					CBr = math.sqrt(CBrsq)
					CBr3 = math.sqrt(CBrsq + CBz ** 2)
					theta = math.atan2(CBy, CBx)
					if abs(theta) > self.POINT_THETA_ABSMAX:
						rospy.logwarn(f"Cannot point at object '{object}' at position ({Bx:.3f}, {By:.3f}, {Bz:.3f}) due to excessive theta {abs(theta):.3f} > {self.POINT_THETA_ABSMAX:.3f}")
					elif CBr3 <= self.POINT_NOMINAL_DIST:
						rospy.logwarn(f"Cannot point at object '{object}' at position ({Bx:.3f}, {By:.3f}, {Bz:.3f}) as it is closer to the control point than the nominal pointing distance {self.POINT_NOMINAL_DIST:.3f}")
					else:
						R = self.POINT_NOMINAL_DIST
						beta = self.POINT_FINGER_ANGLE
						cbeta = math.cos(beta)
						sbeta = math.sin(beta)
						alpha = math.atan2(-CBz, CBr) - math.asin(min(max(sbeta * R / CBr3, -1), 1))
						gamma = alpha + beta
						if gamma <= 0 or gamma >= self.POINT_GAMMA_MAX:
							rospy.logwarn(f"Cannot point at object '{object}' at position ({Bx:.3f}, {By:.3f}, {Bz:.3f}) due to gamma {gamma:.3f} out of range ({0:.3f},{self.POINT_GAMMA_MAX:.3f})")
						else:
							F = self.POINT_DIM_FINGER
							H = self.POINT_DIM_HEIGHT
							cgamma = math.cos(gamma)
							sgamma = math.sin(gamma)
							calpha = math.cos(alpha)
							salpha = math.sin(alpha)
							Dr = CBr - R * cgamma + F * calpha - H * salpha
							Dz = Bz + R * sgamma - F * salpha - H * calpha
							Drmax = self.POINT_REACH_MAX
							Dzmin = Tz + self.POINT_TABLE_CLEARANCE
							lambdar = (Dr - Drmax) / cgamma
							lambdaz = (Dzmin - Dz) / sgamma
							if lambdar > 0 or lambdaz > 0:
								if lambdar >= lambdaz:  # Implies lambdar > 0
									R += lambdar  # Note: From the geometrical construction method of gamma it is almost impossible that the relocated D point is higher than the control point, and for non-pathological cases is usually significantly below
								else:  # Implies lambdaz > 0 and Dz < Dzmin
									Dz = Dzmin
									if Dr > Drmax:
										Dr = Drmax
									BDr = Dr - CBr
									BDz = Dz - Bz
									K = F * sbeta - H * cbeta
									J = R - F * cbeta - H * sbeta
									lambdarz = math.sqrt(BDr ** 2 + BDz ** 2 - K ** 2) - J
									psirz = math.atan2(BDz, -BDr) - gamma - math.atan2(K, J + lambdarz)
									if lambdarz < -self.POINT_CHECK_TOL or psirz * R < -self.POINT_CHECK_TOL:
										rospy.logwarn(f"Cannot point at object '{object}' at position ({Bx:.3f}, {By:.3f}, {Bz:.3f}) because lambdarz {lambdarz:.3g} and/or psirz {psirz:.3g} are not positive with a tolerance scale of {self.POINT_CHECK_TOL:.3g}")
										R = gamma = None
									else:
										R += max(lambdarz, 0)
										gamma += max(psirz, 0)
										if gamma <= 0 or gamma >= self.POINT_GAMMA_MAX:
											rospy.logwarn(f"Cannot point at object '{object}' at position ({Bx:.3f}, {By:.3f}, {Bz:.3f}) due to adjusted gamma {gamma:.3f} out of range ({0:.3f},{self.POINT_GAMMA_MAX:.3f})")
											R = gamma = None
							if R is not None and gamma is not None:
								alpha = gamma - beta
								cgamma = math.cos(gamma)
								sgamma = math.sin(gamma)
								Lr = CBr - R * cgamma
								Lz = Bz + R * sgamma
								pi4 = math.pi / 4
								root2 = math.sqrt(2)
								angle1 = pi4 + (theta + alpha) / 2
								angle2 = pi4 + (theta - alpha) / 2
								ctheta = math.cos(theta)
								stheta = math.sin(theta)
								Or, Ot, Ox, Oy = self.POINT_OFFSET_R if arm_control.is_right else self.POINT_OFFSET_L
								Ot *= arm_control.arm_sign
								Oy *= arm_control.arm_sign
								arm_target = geometry_msgs.msg.Pose(
									position=geometry_msgs.msg.Point(
										x=Cx + (Lr + Or) * ctheta - Ot * stheta + Ox,
										y=Cy + (Lr + Or) * stheta + Ot * ctheta + Oy,
										z=Lz,
									),
									orientation=geometry_msgs.msg.Quaternion(w=math.cos(angle1) / root2, x=-math.sin(angle2) / root2, y=math.cos(angle2) / root2, z=math.sin(angle1) / root2),
								)
								arm_control.set_pose(hand_target=self.POINT_HAND_TARGET, reset_first=True, stay_time=0, wait_finished=False)
								arm_control.set_pose(arm_target=arm_target, stay_time=2)
			else:
				head_future = COMPLETED_ACTION_FUTURE
				rospy.logwarn(f"Cannot point at unavailable/invalid object '{object}'")
		head_future.wait(wait_finished=True)

	# noinspection PyShadowingBuiltins
	def action_give(self, object):
		detected_object: nicol_demos.msg.DetectedObject = self.current_objects.get(object, None)
		if detected_object:
			head_future = self.perform_look(object, wait_finished=False)
			Bx = detected_object.bottom.x
			By = detected_object.bottom.y
			Bz = detected_object.bottom.z
			if not (0 <= Bx <= self.TABLE_DIM[0] and abs(By) <= self.TABLE_DIM[1] and abs(Bz - self.TABLE_DIM[2]) < 0.01):
				rospy.logwarn(f"Cannot give object '{object}' at position ({Bx:.3f}, {By:.3f}, {Bz:.3f}) due to table dimensions ({self.TABLE_DIM[0]:.3f}, {self.TABLE_DIM[1]:.3f}, {self.TABLE_DIM[2]:.3f})")
			else:
				arm_control = self.get_arm_control(ycoord=By)
				Cx, Cy, Cz = self.GIVE_CONTROL_POINT
				Cy *= arm_control.arm_sign
				CBx = Bx - Cx
				CBy = By - Cy
				CBr = math.sqrt(CBx ** 2 + CBy ** 2)
				theta = math.atan2(CBy, CBx)
				if abs(theta) > self.GIVE_THETA_ABSMAX:
					rospy.logwarn(f"Cannot give object '{object}' at position ({Bx:.3f}, {By:.3f}, {Bz:.3f}) due to excessive theta {abs(theta):.3f} > {self.GIVE_THETA_ABSMAX:.3f}")
				else:
					F = self.GIVE_DIM_FINGER
					H = self.GIVE_DIM_HEIGHT
					FH = math.sqrt(F ** 2 + H ** 2)
					Bzhat = Bz + (F + H + 2 * FH) / 4  # Initial rough estimate of an appropriate Bz
					alphahat = math.atan2(Cz - Bzhat, CBr)
					Bzhat = Bz + H * math.cos(alphahat) + F * math.sin(alphahat)
					alpha = math.atan2(Cz - Bzhat, CBr)
					if alpha < 0 or alpha > self.GIVE_ALPHA_MAX:
						rospy.logwarn(f"Cannot give object '{object}' at position ({Bx:.3f}, {By:.3f}, {Bz:.3f}) due to alpha {alpha:.3f} out of range [0,{self.GIVE_ALPHA_MAX:.3f}]")
					else:
						Bz = Bz + H * math.cos(alpha) + F * math.sin(alpha)
						Or, Ot, Ox, Oy = self.GIVE_OFFSET_R if arm_control.is_right else self.GIVE_OFFSET_L
						Ot *= arm_control.arm_sign
						Oy *= arm_control.arm_sign
						ctheta = math.cos(theta)
						stheta = math.sin(theta)
						pi4 = math.pi / 4
						root2 = math.sqrt(2)
						angle1 = pi4 + (theta + alpha) / 2
						angle2 = pi4 + (theta - alpha) / 2
						point_c = geometry_msgs.msg.Pose(
							position=geometry_msgs.msg.Point(x=Cx + (CBr + Or) * ctheta - Ot * stheta + Ox, y=Cy + (CBr + Or) * stheta + Ot * ctheta + Oy, z=Bz),
							orientation=geometry_msgs.msg.Quaternion(w=math.cos(angle1) / root2, x=-math.sin(angle2) / root2, y=math.cos(angle2) / root2, z=math.sin(angle1) / root2),
						)
						point_d = geometry_msgs.msg.Pose(
							position=geometry_msgs.msg.Point(x=point_c.position.x - 0.5 * self.GIVE_STROKE * ctheta, y=point_c.position.y - 0.5 * self.GIVE_STROKE * stheta, z=point_c.position.z + self.GIVE_DESCENT),
							orientation=point_c.orientation,
						)
						point_bb = geometry_msgs.msg.Pose(
							position=geometry_msgs.msg.Point(x=point_c.position.x - 0.5 * self.GIVE_STROKE * ctheta, y=point_c.position.y - 0.5 * self.GIVE_STROKE * stheta, z=point_c.position.z + 0.5 * self.GIVE_CLEARANCE),
							orientation=point_c.orientation,
						)
						point_b = geometry_msgs.msg.Pose(
							position=geometry_msgs.msg.Point(x=point_c.position.x - self.GIVE_STROKE * ctheta, y=point_c.position.y - self.GIVE_STROKE * stheta, z=point_c.position.z + self.GIVE_CLEARANCE),
							orientation=point_c.orientation,
						)
						point_a = geometry_msgs.msg.Pose(
							position=geometry_msgs.msg.Point(x=point_b.position.x, y=point_b.position.y, z=point_b.position.z + self.GIVE_DESCENT),
							orientation=point_b.orientation,
						)
						arm_control.close_hand(reset_first=True, stay_time=0, wait_finished=False)
						arm_control.set_pose(arm_target=point_a, stay_time=0, wait_finished=False)
						arm_control.set_pose(arm_target=point_b, stay_time=0, wait_finished=False)
						arm_control.set_pose(hand_target=self.GIVE_HAND_TARGET, stay_time=0, wait_finished=False)
						arm_control.set_pose(arm_target=point_bb, stay_time=0, wait_finished=False)
						arm_control.set_pose(arm_target=point_c, stay_time=0, wait_finished=False)
						arm_control.open_hand(stay_time=2, wait_finished=False)
						arm_control.set_pose(arm_target=point_d, stay_time=0)
			head_future.wait(wait_finished=True)
		else:
			rospy.logwarn(f"Cannot give unavailable/invalid object '{object}'")

	def action_express(self, emotion):
		if emotion == 'emotion':
			emotion = FacialExpression.EXPRESSION_HAPPY
		self.facial_expression.set_expression(emotion, stay_time=self.EMOTION_STAY_TIME)

	def get_arm_control(self, ycoord):
		ysign = 1 if ycoord >= 0 else -1
		return next(arm_control for arm_control in self.arm_controls if arm_control.arm_sign == ysign)

	### Run thread ###

	def step(self):
		self.facial_expression.step()
		self.head_control.step()
		self.arm_control_l.step()
		self.arm_control_r.step()

#
# Miscellaneous
#

# Capitalize the first letter of a string without touching the rest
def capitalize(text):
	return text[0].upper() + text[1:]

# Strip whitespace and leading punctuation, and collapse to empty string if on alphanumeric characters are present
def clean_text(text):
	if not any(char.isalnum() for char in text):
		return ''
	else:
		return re.sub(r'^[\s.?!,:;\-<>]*(.*\S)\s*$', r'\1', text)

# Format a tuple
def format_tuple(tup, fmt):
	return f"({', '.join(format(value, fmt) for value in tup)})"

# Decode escape characters in a string (e.g. '\\n' --> '\n')
ESCAPE_SEQUENCE_RE = re.compile(r'''
	( \\U........      # 8-digit hex escapes
	| \\u....          # 4-digit hex escapes
	| \\x..            # 2-digit hex escapes
	| \\[0-7]{1,3}     # Octal escapes
	| \\N\{[^}]+}      # Unicode characters by name
	| \\[\\'"abfnrtv]  # Single-character escapes
	)''', re.VERBOSE)
def decode_escapes(text):
	def decode_match(match):
		return codecs.decode(match.group(0), 'unicode-escape')
	return ESCAPE_SEQUENCE_RE.sub(decode_match, text)

#
# Main
#

# Run main function
if __name__ == "__main__":
	main()
# EOF
