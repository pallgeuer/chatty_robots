# ROS action server that supports multiple parallel requests
# Author: Philipp Allgeuer

# Imports
import genpy
import actionlib
import actionlib_msgs.msg

# Constants
ActionGoalMsg = genpy.Message      # Goal message type placeholder (e.g. PACKAGE.msg.NAMEGoal)
ActionResultMsg = genpy.Message    # Result message type placeholder (e.g. PACKAGE.msg.NAMEResult)
ActionFeedbackMsg = genpy.Message  # Feedback message type placeholder (e.g. PACKAGE.msg.NAMEFeedback)

# Action info class
class ActionInfo:

	### Any thread ###

	def __init__(self, goal: ActionGoalMsg, handle: actionlib.ServerGoalHandle):
		# Note: It is not critically important to make sure the ActionInfo and ServerGoalHandle go out of scope as soon as possible
		self.goal = goal
		self.handle = handle
		with self.handle.action_server.lock:
			self.handle.status_tracker.handle_wrapper = self  # We inject into the status tracker (as it uniquely identifies a goal while a handle doesn't) a reference to the corresponding ActionInfo
			self.id = self.handle.get_goal_id().id

	def get_id(self) -> str:
		return self.id

	def feedback(self, feedback: ActionFeedbackMsg):
		self.handle.publish_feedback(feedback)

	def canceled(self, result: ActionResultMsg = None, status: str = ""):
		# Only call this method (a single time) if True was returned from cancel_cb() for this ActionInfo
		self.handle.set_canceled(result=result, text=status)

	def aborted(self, result: ActionResultMsg = None, status: str = ""):
		# Call this if the running goal fails in some way such that the whole goal needs to be aborted
		self.handle.set_aborted(result=result, text=status)

	def completed(self, result: ActionResultMsg, status: str = ""):
		# Call this on successful completion of the running goal
		self.handle.set_succeeded(result=result, text=status)

	def has_terminated(self) -> bool:
		# A goal terminates and can be safely forgotten if:
		#  - goal_cb() with this ActionInfo had False returned (rejected goal)
		#  - cancel_cb() with this ActionInfo had False returned, or returned True and canceled() was subsequently called (canceled goal)
		#  - aborted() was called (aborted goal)
		#  - completed() was called (goal reached)
		return self.handle.get_goal_status().status in (
			actionlib_msgs.msg.GoalStatus.REJECTED,
			actionlib_msgs.msg.GoalStatus.RECALLED,
			actionlib_msgs.msg.GoalStatus.PREEMPTED,
			actionlib_msgs.msg.GoalStatus.ABORTED,
			actionlib_msgs.msg.GoalStatus.SUCCEEDED,
			actionlib_msgs.msg.GoalStatus.LOST,
		)

# Multi-action server class
class MultiActionServer:

	### Main thread ###

	def __init__(self, ns, ActionSpec, goal_cb=None, cancel_cb=None):
		# ns = Namespace in which to access the action (e.g. the goal topic is ns/goal)
		# ActionSpec = Action message type (e.g. PACKAGE.msg.NAMEAction)
		# goal_cb = Goal callback that gets called when a new goal is received (Callable[[ActionInfo], Union[bool, tuple[bool], tuple[bool, str], tuple[bool, str, ActionResultMsg]] = Return whether the goal was accepted and started, and possibly a reason, and result (if rejected). You need to store the ActionInfo in order to be able to call completed(), or other methods, on it later)
		# cancel_cb = Cancel callback that gets called when a goal cancellation is received (Callable[[ActionInfo], Union[bool, tuple[bool], tuple[bool, str], tuple[bool, str, ActionResultMsg]] = Return whether the cancellation is already complete (and possibly a reason and result), or is still pending and will be ended by a single explicit call to ActionInfo.canceled() sometime in the future. Normal case is to never call ActionInfo.canceled() as it will be called automatically.)
		self.ns = ns
		self.ActionSpec = ActionSpec
		self.goal_cb = goal_cb
		self.cancel_cb = cancel_cb
		self.action_server = actionlib.ActionServer(self.ns, self.ActionSpec, goal_cb=self._handle_goal, cancel_cb=self._handle_cancel, auto_start=False)

	def start(self):
		self.action_server.start()
		return self

	def stop(self):
		self.action_server.stop()
		self.action_server = None

	### Callback threads ###

	def _handle_goal(self, handle: actionlib.ServerGoalHandle):
		# Note: Action server lock is locked during this method

		goal = handle.get_goal()
		assert goal is not None

		reason = ""
		result = None
		decision = self.goal_cb(ActionInfo(goal, handle))
		if isinstance(decision, tuple):
			num_vars = len(decision)
			if num_vars == 3:
				accept, reason, result = decision
			elif num_vars == 2:
				accept, reason = decision
			elif num_vars == 1:
				accept, = decision
			else:
				raise ValueError(f"Invalid goal callback decision tuple: {decision}")
		else:
			accept = decision

		if accept:
			handle.set_accepted(text=reason)
		else:
			handle.set_rejected(result=result, text=reason)

	def _handle_cancel(self, handle: actionlib.ServerGoalHandle):
		# Note: Action server lock is locked during this method

		action_info: ActionInfo = handle.status_tracker.handle_wrapper  # Note: This should always exist by the time of a cancel callback (if not, there is some logic error)

		status = ""
		result = None
		cancellation = self.cancel_cb(action_info)
		if isinstance(cancellation, tuple):
			num_vars = len(cancellation)
			if num_vars == 3:
				delayed, status, result = cancellation
			elif num_vars == 2:
				delayed, status = cancellation
			elif num_vars == 1:
				delayed, = cancellation
			else:
				raise ValueError(f"Invalid cancel callback decision tuple: {cancellation}")
		else:
			delayed = cancellation

		if not delayed:
			action_info.canceled(result=result, status=status)
# EOF
