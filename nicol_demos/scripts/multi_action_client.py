# ROS action client that supports multiple parallel requests
# Author: Philipp Allgeuer

# Imports
from __future__ import annotations
import threading
from typing import Optional, Callable
import rospy
import genpy
import actionlib
from actionlib import CommState
from actionlib.simple_action_client import SimpleGoalState as GoalState
import actionlib_msgs.msg

# Fix actionlib.action_client.get_name_of_constant() to not get GoalStatus.PENDING wrong
def get_name_of_constant(C, n):
	for k, v in C.__dict__.items():
		if k.isupper() and isinstance(v, int) and v == n:  # Note: Added check that key is upper case
			return k
	return "NO_SUCH_STATE_%d" % n
actionlib_msgs.msg.GoalStatus.to_string = classmethod(get_name_of_constant)

# Constants
ActionGoalMsg = genpy.Message      # Goal message type placeholder (e.g. PACKAGE.msg.NAMEGoal)
ActionResultMsg = genpy.Message    # Result message type placeholder (e.g. PACKAGE.msg.NAMEResult)
ActionFeedbackMsg = genpy.Message  # Feedback message type placeholder (e.g. PACKAGE.msg.NAMEFeedback)

# Action goal class
class ActionGoal:

	### Main thread ###

	def __init__(
		self,
		goal: ActionGoalMsg,
		handle: actionlib.ClientGoalHandle,
		active_cb: Optional[Callable[[str], None]],
		done_cb: Optional[Callable[[str, int, Optional[ActionResultMsg]], None]],
		feedback_cb: Optional[Callable[[str, ActionFeedbackMsg], None]],
	):
		# Note: Action client safely forgets/disconnects from a goal when the corresponding ClientGoalHandle (self.handle) is deleted/garbage collected
		#       (more precisely when the CommStateMachine inside it is). Thus, it is safe to just let an ActionGoal go out of scope.
		self.goal = goal  # Goal message
		self.handle = handle  # Action client safely forgets/disconnects from a goal when this object is deleted/garbage collected (more precisely the CommStateMachine inside it)
		self.handle.comm_state_machine.handle_wrapper = self  # We inject into the comm state machine (as it uniquely identifies a goal while a handle doesn't) a reference to the corresponding ActionGoal
		self.id = self.handle.comm_state_machine.action_goal.goal_id.id  # Goal ID that is passed to callbacks
		self.active_cb = active_cb  # --> active_cb(goal_id)
		self.done_cb = done_cb  # --> done_cb(goal_id, goal_status, result) where result may be None
		self.feedback_cb = feedback_cb  # --> feedback_cb(goal_id, feedback)
		self.state = GoalState.PENDING  # Accessed atomically in multiple threads
		self.mutex = threading.Lock()  # This lock is also used for the done condition
		self.done_cond = threading.Condition(lock=self.mutex)  # Used to notify wait_done() when the state transitions to DONE in a callback thread

	def disconnect(self):
		# This can optionally be used prior to ActionGoal's going out of scope to make them get deleted faster by breaking a reference cycle (but no other method can be called after this).
		# Note that the internal ClientGoalHandle and CommStateMachine though get deleted immediately due to zero reference count as soon as the handle is set to None below.
		with self.mutex:
			self.handle = None

	def get_id(self) -> str:
		return self.id

	def get_state(self) -> int:
		# Values defined in: GoalState = actionlib.simple_action_client.SimpleGoalState
		# 0 = PENDING = Goal is pending or recalling
		# 1 = ACTIVE  = Goal is active or preempting
		# 2 = DONE    = Goal is complete and has a result available
		return self.state

	def get_state_text(self) -> str:
		# noinspection PyUnresolvedReferences
		return GoalState.to_string(self.state)

	def get_goal_status(self) -> int:
		# Refer to: http://wiki.ros.org/actionlib/DetailedDescription#Server_Description
		# Values defined in: actionlib_msgs.msg.GoalStatus
		# 0 = PENDING    = The goal has yet to be processed by the action server
		# 1 = ACTIVE     = The goal is currently being processed by the action server
		# 2 = PREEMPTED  = The goal received a cancel request after it started executing and has since completed its execution (Terminal State)
		# 3 = SUCCEEDED  = The goal was achieved successfully by the action server (Terminal State)
		# 4 = ABORTED    = The goal was aborted during execution by the action server due to some failure (Terminal State)
		# 5 = REJECTED   = The goal was rejected by the action server without being processed, because the goal was unattainable or invalid (Terminal State)
		# 6 = PREEMPTING = The goal received a cancel request after it started executing and has not yet completed execution
		# 7 = RECALLING  = The goal received a cancel request before it started executing, but the action server has not yet confirmed that the goal is canceled
		# 8 = RECALLED   = The goal received a cancel request before it started executing and was successfully cancelled (Terminal State)
		# 9 = LOST       = An action client can determine that a goal is LOST (this should not be sent over the wire by an action server)
		return self.handle.get_goal_status()

	def get_goal_status_text(self) -> str:
		return self.handle.get_goal_status_text()

	def is_done(self) -> bool:
		return self.state == GoalState.DONE

	def wait_done(self, done_timeout=None, auto_cancel=False, wait_cancel=True, cancel_timeout=None) -> bool:
		if auto_cancel:
			if self.wait_done(done_timeout=done_timeout):
				return True
			else:
				return self.cancel(wait_done=wait_cancel, done_timeout=cancel_timeout)
		elif self.state == GoalState.DONE:
			return True
		else:
			# noinspection PyTypeChecker
			loop_period = rospy.Duration(0.1)
			zero_duration = rospy.Duration(0)
			done_timeout = rospy.Duration() if done_timeout is None else done_timeout if isinstance(done_timeout, rospy.Duration) else rospy.Duration(done_timeout)
			timeout_time = None if done_timeout.is_zero() else rospy.get_rostime() + done_timeout
			with self.done_cond:
				while self.state != GoalState.DONE and not rospy.is_shutdown():
					if timeout_time is None:
						time_left = loop_period
					else:
						time_left = timeout_time - rospy.get_rostime()
						if time_left <= zero_duration:
							break
						elif time_left > loop_period:
							time_left = loop_period
					self.done_cond.wait(time_left.to_sec())
			return self.state == GoalState.DONE

	def cancel(self, wait_done=False, done_timeout=None) -> bool:
		self.handle.cancel()
		if wait_done:
			return self.wait_done(done_timeout=done_timeout)
		else:
			return self.state == GoalState.DONE

	def get_result(self) -> Optional[ActionResultMsg]:
		return self.handle.get_result()

# Multi-action client class
class MultiActionClient:

	### Main thread ###

	def __init__(self, ns, ActionSpec, active_cb=None, done_cb=None, feedback_cb=None):
		# ns = Namespace in which to access the action (e.g. the goal topic is ns/goal)
		# ActionSpec = Action message type (e.g. PACKAGE.msg.NAMEAction)
		# active_cb = Default goal callback that gets called on transitions to GoalState.ACTIVE (None = No callback)
		# done_cb = Default goal callback that gets called on transitions to GoalState.DONE (None = No callback)
		# feedback_cb = Default goal callback that gets called when feedback is received (None = No callback)
		self.ns = ns
		self.ActionSpec = ActionSpec
		self.active_cb = active_cb
		self.done_cb = done_cb
		self.feedback_cb = feedback_cb
		self.handle_mutex = threading.Lock()
		self.action_client = actionlib.ActionClient(self.ns, self.ActionSpec)

	def stop(self):
		self.action_client.stop()
		self.action_client = None

	def wait_for_server(self, timeout=None) -> bool:
		return self.action_client.wait_for_server(rospy.Duration() if timeout is None else timeout if isinstance(timeout, rospy.Duration) else rospy.Duration(timeout))

	def send_goal(self, goal, wait_done=False, done_timeout=None, auto_cancel=False, wait_cancel=True, cancel_timeout=None, active_cb=None, done_cb=None, feedback_cb=None) -> ActionGoal:
		# goal = Goal message to send (e.g. of type PACKAGE.msg.NAMEGoal)
		# wait_done = Whether to block until the goal is done
		# done_timeout = If waiting to be done, the timeout to use (0 = No timeout)
		# auto_cancel = If waiting to be done, whether to automatically cancel the goal if the done timeout expires
		# wait_cancel = If waiting to be done and auto-cancel, whether to also wait for a possible cancel to finish
		# cancel_timeout = If waiting to be done and auto-cancel, the timeout to use for waiting for the cancel (0 = No timeout)
		# active_cb = Goal callback that gets called on transitions to GoalState.ACTIVE (use default callback if None)
		# done_cb = Goal callback that gets called on transitions to GoalState.DONE (use default callback if None)
		# feedback_cb = Goal callback that gets called when feedback is received (use default callback if None)
		# Note: All callbacks are called from possibly different background threads, but never concurrently due to a mutex, and the feedback callback is never called after the done callback.
		with self.handle_mutex:
			action_goal = ActionGoal(
				goal=goal,
				handle=self.action_client.send_goal(goal, transition_cb=self._handle_transition, feedback_cb=self._handle_feedback),
				active_cb=active_cb or self.active_cb,
				done_cb=done_cb or self.done_cb,
				feedback_cb=feedback_cb or self.feedback_cb,
			)
		if wait_done:
			action_goal.wait_done(done_timeout=done_timeout, auto_cancel=auto_cancel, wait_cancel=wait_cancel, cancel_timeout=cancel_timeout)
		return action_goal

	def cancel_all_goals(self):
		self.action_client.cancel_all_goals()

	def cancel_goals_at_and_before_time(self, time):
		self.action_client.cancel_goals_at_and_before_time(time)

	### Callback threads ###

	def _handle_transition(self, handle: actionlib.ClientGoalHandle):
		with self.handle_mutex:
			action_goal: ActionGoal = handle.comm_state_machine.handle_wrapper
		with action_goal.mutex:
			if action_goal.handle is None:
				return
			comm_state = handle.get_comm_state()
			# noinspection PyUnresolvedReferences
			error_msg = f"Received comm state {CommState.to_string(comm_state)} while in action state {GoalState.to_string(action_goal.state)} with {self.__class__.__name__} in namespace {rospy.resolve_name(self.ns)}"
			if comm_state == CommState.ACTIVE:
				if action_goal.state == GoalState.PENDING:
					action_goal.state = GoalState.ACTIVE
					if action_goal.active_cb:
						action_goal.active_cb(action_goal.id)
				elif action_goal.state == GoalState.DONE:
					rospy.logerr(error_msg)
			elif comm_state == CommState.RECALLING:
				if action_goal.state != GoalState.PENDING:
					rospy.logerr(error_msg)
			elif comm_state == CommState.PREEMPTING:
				if action_goal.state == GoalState.PENDING:
					action_goal.state = GoalState.ACTIVE
					if action_goal.active_cb:
						action_goal.active_cb(action_goal.id)
				elif action_goal.state == GoalState.DONE:
					rospy.logerr(error_msg)
			elif comm_state == CommState.DONE:
				if action_goal.state in (GoalState.PENDING, GoalState.ACTIVE):
					if action_goal.done_cb:
						action_goal.done_cb(action_goal.id, handle.get_goal_status(), handle.get_result())
					action_goal.state = GoalState.DONE
					action_goal.done_cond.notify_all()
				elif action_goal.state == GoalState.DONE:
					rospy.logerr(error_msg)

	def _handle_feedback(self, handle: actionlib.ClientGoalHandle, feedback: ActionFeedbackMsg):
		with self.handle_mutex:
			action_goal: ActionGoal = handle.comm_state_machine.handle_wrapper
		with action_goal.mutex:
			if action_goal.handle is None or action_goal.state == GoalState.DONE:
				return
			if action_goal.feedback_cb:
				action_goal.feedback_cb(action_goal.id, feedback)
# EOF
