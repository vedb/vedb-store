""""""

class Subject(object):
    """ Base class for Subject Object. """

    def __init__(self, main_directory):
        """ Constructor.

        Parameters
        ----------
        parameter_1: str
            Parameter_1 .

        """
        self.subject_id = None
        self.subject_age = None
        self.subject_gender = None
        self.subject_ethnicity = None 

        self.experimenter = None # ID, not name
        self.university = None # UNR / Bates / NDSU
        self.scene_category = None #(outdoor / indoor) ## (go here)
        self.task = None ## (do this) pp best in snippets only * activities # Sub-labels for time points, based on manual labeling or analysis
        self.recording_duration = None
        self.main_directory = main_directory # Name for parent folder for session
        self.system = None
