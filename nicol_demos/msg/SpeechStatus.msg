uint8 CACHED=0
uint8 PLAYING=1
uint8 PLAYED=2
uint8 CLEARED=3

uint64 speech_id  # Which speech request this status update refers to
uint8 event       # Which event type has occurred (e.g. FINISHED)
uint16[] items    # Which items of the speech request are affected
