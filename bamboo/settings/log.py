# bamboo/settings/log.py
class LoggerConfig:
    LOGGING_ENABLED = True

# Getter and setter functions
def is_logging_enabled():
    return LoggerConfig.LOGGING_ENABLED

def set_logging(enabled):
    LoggerConfig.LOGGING_ENABLED = enabled