uint8 NEW_CHAT=0  # A new chat has just started
uint8 USER=1      # The user has spoken
uint8 ROBOT=2     # The robot has spoken

Header header     # Header with timestamp
uint8 speaker     # Speaker of text (e.g. USER, ROBOT or NEW_CHAT)
string[] texts    # Spoken texts (empty if NEW_CHAT)
