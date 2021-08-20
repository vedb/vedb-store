
from data_base_tools.session import Session
#from data_base_tools.recorded_data import RecordedData
from data_base_tools.subject import Subject
import numpy as np
import cv2

print("Hello!!!")

main_directory = '/hdd01/kamran_sync/vedb/recordings_pilot/pilot_study_1/'
session_id = '423/'
my_session = Session(session_id, main_directory, move = False, export_directory = '/000/')

print('created session for: ',my_session.session_id)

print('export directory: ', my_session.export_directory)

my_session.from_pupil()