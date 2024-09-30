uint8 EVENT=0     # Instantaneous item event has occurred
uint8 STARTED=1   # Interval item has started
uint8 FINISHED=2  # Interval item has finished

Header header     # Header with timestamp
string item       # Item that has had a status change (e.g. 'user_text', 'say', 'action_look', 'reset_neutral', 'reset_chat')
uint8 status      # Status of the item (e.g. FINISHED)
