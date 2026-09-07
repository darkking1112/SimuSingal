"""Independent simulation workspace; no signal asset tables."""
from common.storage import Workspace as RunWorkspace

class Workspace(RunWorkspace):
    project = "communication_sim"
