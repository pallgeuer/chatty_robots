#!/usr/bin/env python3
# Author: Philipp Allgeuer
# Automatic speech recognition client
# Usage: rosrun nicol_demos speech_asr_client.py [PARAMS]
# Where PARAMS is anything like _verbose:=false _detect:=false _live_text:=false _max_duration:=60 _done_timeout:=90 [CAREFUL: These are sticky once set even a single time!]

# Imports
from enum import Enum
import rospy
import actionlib_msgs.msg
import nicol_demos.msg
import multi_action_client

# Stop reason enumeration
class StopReason(Enum):
	UNKNOWN = nicol_demos.msg.PerformASRResult.STOP_UNKNOWN
	TIMEOUT = nicol_demos.msg.PerformASRResult.STOP_TIMEOUT
	DETECTED = nicol_demos.msg.PerformASRResult.STOP_DETECTED
	DURATION = nicol_demos.msg.PerformASRResult.STOP_DURATION
	REQUEST = nicol_demos.msg.PerformASRResult.STOP_REQUEST

# Speech ASR client class (can subclass this for custom callback handling)
class SpeechASRClient(multi_action_client.MultiActionClient):

	### Main thread ###

	def __init__(self, ns='speech_asr', verbose=False):
		super().__init__(ns=ns, ActionSpec=nicol_demos.msg.PerformASRAction, active_cb=self.active_cb, done_cb=self.done_cb, feedback_cb=self.feedback_cb)
		self.verbose = verbose

	def perform_asr(self, detect_start=True, detect_stop=True, start_timeout=0.0, min_duration=3.0, max_duration=30.0, min_period=3.0, live_text=False, done_timeout=None) -> str:
		goal = nicol_demos.msg.PerformASRGoal(detect_start=detect_start, detect_stop=detect_stop, start_timeout=start_timeout, min_duration=min_duration, max_duration=max_duration, min_period=min_period, live_text=live_text)
		action_goal = self.send_goal(goal=goal, wait_done=True, done_timeout=done_timeout, auto_cancel=True, wait_cancel=True)
		result: nicol_demos.msg.PerformASRResult = action_goal.get_result()  # noqa
		return result.text if result is not None else ""

	def perform_asr_async(self, detect_start=True, detect_stop=True, start_timeout=0.0, min_duration=3.0, max_duration=30.0, min_period=3.0, live_text=False, wait_done=False, done_timeout=None, auto_cancel=False, wait_cancel=True, cancel_timeout=None, active_cb=None, done_cb=None, feedback_cb=None) -> multi_action_client.ActionGoal:
		goal = nicol_demos.msg.PerformASRGoal(detect_start=detect_start, detect_stop=detect_stop, start_timeout=start_timeout, min_duration=min_duration, max_duration=max_duration, min_period=min_period, live_text=live_text)
		action_goal = self.send_goal(goal=goal, wait_done=wait_done, done_timeout=done_timeout, auto_cancel=auto_cancel, wait_cancel=wait_cancel, cancel_timeout=cancel_timeout, active_cb=active_cb, done_cb=done_cb, feedback_cb=feedback_cb)
		return action_goal

	### Callback threads ###

	def active_cb(self, goal_id: str):
		if self.verbose:
			rospy.loginfo(f"\033[34mAction {goal_id} has ACTIVATED\033[0m")

	def feedback_cb(self, goal_id: str, feedback: nicol_demos.msg.PerformASRFeedback):
		if self.verbose:
			text_info = f"with live text: \"{feedback.cur_text if len(feedback.cur_text) <= 60 else '...' + feedback.cur_text[-57:]}\"" if feedback.have_text else "without live text"
			rospy.loginfo(f"\033[34mAction {goal_id} is ONGOING: Listening for {feedback.cur_listened:.3f}s, recording for {feedback.cur_recorded:.3f}s ({'STARTED' if feedback.started else 'WAITING'}) {text_info}\033[0m")

	def done_cb(self, goal_id, goal_status, result):
		if self.verbose:
			try:
				stop_reason = StopReason(result.stop_reason).name
			except ValueError:
				stop_reason = 'INVALID REASON'
			# noinspection PyUnresolvedReferences
			rospy.loginfo(f"\033[34mAction {goal_id} has FINISHED ({actionlib_msgs.msg.GoalStatus.to_string(goal_status)} due to stop reason {stop_reason}): Listened for {result.listened:.3f}s, recorded for {result.recorded:.3f}s ({'STARTED' if result.started else 'NEVER STARTED'}) with final text: \"{result.text}\"\033[0m")

# Main function
def main():

	rospy.init_node('speech_asr_client')

	verbose = rospy.get_param('~verbose', True)
	detect_start = rospy.get_param('~detect_start', True)
	detect_stop = rospy.get_param('~detect_stop', True)
	if (detect := rospy.get_param('~detect', None)) is not None:
		detect_start = detect
		detect_stop = detect
	start_timeout = rospy.get_param('~start_timeout', 0.0)
	min_duration = rospy.get_param('~min_duration', 3.0)
	max_duration = rospy.get_param('~max_duration', 30.0)
	min_period = rospy.get_param('~min_period', 3.0)
	live_text = rospy.get_param('~live_text', True)
	done_timeout = rospy.get_param('~done_timeout', 0.0)

	client = SpeechASRClient(verbose=verbose)
	rospy.loginfo(f"Waiting for action server to exist: {client.ns}")
	client.wait_for_server()

	rospy.loginfo("Performing ASR using action server...")
	text = client.perform_asr(detect_start=detect_start, detect_stop=detect_stop, start_timeout=start_timeout, min_duration=min_duration, max_duration=max_duration, min_period=min_period, live_text=live_text, done_timeout=done_timeout)
	rospy.loginfo(f"ASR text: \"{text}\"")

# Run main function
if __name__ == "__main__":
	main()
# EOF
