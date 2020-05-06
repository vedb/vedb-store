import yaml
import sys
import os
import cv2

from data_base_tools.subject import Subject

class Session:
    """ Base class for Session Object. """

    def __init__(self, session_id, main_directory, move = False, export_directory = '/000/'):
        """ Constructor.

        Parameters
        ----------
        main_directory: str
            Path to the recording main directory.

        """
        self.session_id = session_id
        self.main_directory = main_directory
        self.move = move
        self.export_directory = export_directory
        with open("/home/kamran/Code/pupil_recording_interface/data_collection.yml") as f:
            self.config_file = yaml.load(f, Loader = yaml.FullLoader)



    def from_pupil(self):

        # Session metadata

        self.subject = Subject(self.session_id)
        self.world = RecordedData(self.session_id, self.main_directory, data_type = 'world')
        self.eyeR = RecordedData(self.session_id, self.main_directory, data_type = 'eye0')
        self.eyeL = RecordedData(self.session_id, self.main_directory, data_type = 'eye1')
        self.gaze = RecordedData(self.session_id, self.main_directory, data_type = 'gaze')
        self.head = RecordedData(self.session_id, self.main_directory, data_type = 't265')

        print('Data Collection Settings: \n')
        for k in self.config_file['Experiment'].keys():
            print(k, ' : ', self.config_file['Experiment'][k])
        print('\n')

        # recording folder
        recording_directory = self.config_file['Experiment']['recording_directory']


        print('\nSystem Settings:')
        for k in self.config_file['System'].keys():
            print(k, ' : ', self.config_file['System'][k])
        print('\n')


        return None


    def save(self, directory_path):
        return True

    # Todo: This should not be here, most probably goes into vm_tools
    def calibrate_gaze(cyclopeanPOR_XY, truePOR_XY, method = cv2.RANSAC, threshold = 5, plottingFlag = False):

        result = cv2.findHomography(cyclopeanPOR_XY, truePOR_XY, method = method , ransacReprojThreshold = threshold)
        #print(result[0])
        #print('size', len(result[1]),'H=', result[1])
        totalFrameNumber = truePOR_XY.shape[0]
        arrayOfOnes = np.ones((totalFrameNumber,1), dtype = float)

        homogrophy = result[0]
        print('H=', homogrophy, '\n')
        #print('Res', result[1])
        cyclopeanPOR_XY = np.hstack((cyclopeanPOR_XY, arrayOfOnes))
        truePOR_XY = np.hstack((truePOR_XY, arrayOfOnes))
        projectedPOR_XY = np.zeros((totalFrameNumber,3))
        
        for i in range(totalFrameNumber):
            projectedPOR_XY[i,:] = np.dot(homogrophy, cyclopeanPOR_XY[i,:])
            #print cyclopeanPOR_XY[i,:]
        
        #projectedPOR_XY[:, 0], projectedPOR_XY[:, 1] = metricToPixels(projectedPOR_XY[:, 0], projectedPOR_XY[:, 1])
        #cyclopeanPOR_XY[:, 0], cyclopeanPOR_XY[:, 1] = metricToPixels(cyclopeanPOR_XY[:, 0], cyclopeanPOR_XY[:, 1])
        #truePOR_XY[:, 0], truePOR_XY[:, 1] = metricToPixels(truePOR_XY[:, 0], truePOR_XY[:, 1])
        data = projectedPOR_XY
        frameCount = range(len(cyclopeanPOR_XY))

        if( plottingFlag == True ):
            xmin = 550#min(cyclopeanPOR_XY[frameCount,0])
            xmax = 1350#max(cyclopeanPOR_XY[frameCount,0])
            ymin = 250#min(cyclopeanPOR_XY[frameCount,1])
            ymax = 800#max(cyclopeanPOR_XY[frameCount,1])
            #print xmin, xmax, ymin, ymax
            fig1 = plt.figure(figsize = (10,8))
            plt.plot(data[frameCount,0], data[frameCount,1], 'bx', label='Calibrated POR')
            plt.plot(cyclopeanPOR_XY[frameCount,0], cyclopeanPOR_XY[frameCount,1], 'g2', label='Uncalibrated POR')
            plt.plot(truePOR_XY[frameCount,0], truePOR_XY[frameCount,1], 'r8', label='Ground Truth POR')
            #l1, = plt.plot([],[])
            
            #plt.xlim(xmin, xmax)
            #plt.ylim(ymin, ymax)
            plt.xlabel('X')
            plt.ylabel('Y')
            if ( method == cv2.RANSAC):
                methodTitle = ' RANSAC '
            elif( method == cv2.LMEDS ):
                methodTitle = ' Least Median '
            elif( method == 0 ):
                methodTitle = ' Homography '
            plt.title('Calibration Result using'+ methodTitle+'\nWith System Calibration ')
            plt.grid(True)
            #plt.axis('equal')
            #line_ani = animation.FuncAnimation(fig1, update_line1, frames = 11448, fargs=(sessionData, l1), interval=14, blit=True)
            legend = plt.legend(loc=[0.8,0.92], shadow=True, fontsize='small')# 'upper center'
            plt.show()

        print ('MSE_after = ', findResidualError(projectedPOR_XY, truePOR_XY))
        print ('MSE_before = ', findResidualError(cyclopeanPOR_XY, truePOR_XY))
        return homogrophy

    # Todo: This should not be here, most probably goes into vm_tools
    def plot_fps(self, camera = 'world'):
        import numpy as np
        import matplotlib.pyplot as plt

        #f = np.load("/home/kamran/recordings/flir_test/170/t265_timestamps.npy")
        f = np.load(self.folder + "/" + camera +"_timestamps.npy")
        f = np.diff(f)
        f = f[f!=0]

        fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (12,12))

        axes[0].plot(range(len(f)),1/f, 'ob', markersize = 4, alpha = 0.4)
        axes[0].set_title('FPS Vs. Time', fontsize = 14)
        axes[0].yaxis.grid(True)
        axes[0].xaxis.grid(True)
        axes[0].set_xlabel('# of frames', fontsize = 12)
        axes[0].set_ylabel('FPS', fontsize = 14)


        axes[1].hist(1/f, 100, facecolor = 'g', edgecolor = 'k', linewidth = 1)
        axes[1].set_title('FPS histogram', fontsize = 14)
        axes[1].yaxis.grid(True)
        axes[1].xaxis.grid(True)
        axes[1].set_xlabel('FPS', fontsize = 12)
        axes[1].set_ylabel('count', fontsize = 14)

        fig.suptitle(camera, fontsize = 18)
        #plt.savefig(fname.replace('.hdf','_fps_'+str(fps)+'.png'), dpi=150)
        plt.show()

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def pixels_to_angle_x(array):
        return (array - horizontal_pixels/2) * ratio_x

    def pixels_to_angle_y(array):
        return (array - vertical_pixels/2) * ratio_y

    # Todo: This should not be here, most probably goes into vm_tools
    def plot_gaze_accuracy(self, markerPosition, gazeDataFrame, gazeIndex):

        import pandas as pd
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        import cv2
        from matplotlib import cm
        
        horizontal_pixels = 1280
        vertical_pixels = 1024
        horizontal_FOV = 92.5
        vertical_FOV = 70.8

        ratio_x = horizontal_FOV/horizontal_pixels
        ratio_y = vertical_FOV/vertical_pixels


        gaze_norm_x = gazeDataFrame.iloc[gazeIndex].norm_pos_x.values
        gaze_norm_y = gazeDataFrame.iloc[gazeIndex].norm_pos_y.values

        gaze_pixel_x = gaze_norm_x * horizontal_pixels
        gaze_pixel_y = gaze_norm_y * vertical_pixels

        print('gazeX shape = ', gaze_pixel_x.shape)
        print('gazeY shape = ',gaze_pixel_y.shape)
        #print(np.array([gaze_pixel_x, gaze_pixel_y]).shape)
        gaze_homogeneous = cv2.convertPointsToHomogeneous(np.array([gaze_pixel_x, gaze_pixel_y]).T)
        gaze_homogeneous = np.squeeze(gaze_homogeneous)

        gaze_homogeneous[:,0] = pixels_to_angle_x(gaze_homogeneous[:,0])
        gaze_homogeneous[:,1] = pixels_to_angle_y(gaze_homogeneous[:,1])

        # This is important because the gaze values should be inverted in y direction
        gaze_homogeneous[:,1] = -gaze_homogeneous[:,1]

        print('gaze homogeneous shape =',gaze_homogeneous.shape)

        #print('gaze homogeneous =',gaze_homogeneous[0:5,:])

        marker_homogeneous = cv2.convertPointsToHomogeneous(markerPosition)
        marker_homogeneous = np.squeeze(marker_homogeneous)

        marker_homogeneous[:,0] = pixels_to_angle_x(marker_homogeneous[:,0])
        marker_homogeneous[:,1] = pixels_to_angle_y(marker_homogeneous[:,1])
        print('marker homogeneous shape =',marker_homogeneous.shape)
        #print('marker homogeneous =',marker_homogeneous[0:5,:])


        rmse_x = rmse(marker_homogeneous[:,0], gaze_homogeneous[:,0])
        rmse_y = rmse(marker_homogeneous[:,1], gaze_homogeneous[:,1])
        print('RMSE_az = ', rmse_x)
        print('RMSE_el = ', rmse_y)

        azimuthRange = 45
        elevationRange = 45
        fig = plt.figure(figsize = (10,10))
        plt.plot(marker_homogeneous[:,0], marker_homogeneous[:,1], 'or', markersize = 8, alpha = 0.6, label = 'marker')
        plt.plot(gaze_homogeneous[:,0], gaze_homogeneous[:,1], '+b', markersize = 8, alpha = 0.6, label = 'gaze')
        plt.title('Marker Vs. Gaze Positions (Raw)', fontsize = 18)
        plt.legend(fontsize = 12)
        plt.text(-40,40, ('RMSE_az = %.2f'%(rmse_x)), fontsize = 14)
        plt.text(-40,35, ('RMSE_el = %.2f'%(rmse_y)), fontsize = 14)
        #plt.text(-40,30, ('Distance = %d [inch] %d [cm]'%(depth_inch[depthIndex], depth_cm[depthIndex])), fontsize = 14)
        plt.xlabel('azimuth (degree)', fontsize = 14)
        plt.ylabel('elevation (degree)', fontsize = 14)
        plt.xticks(np.arange(-azimuthRange, azimuthRange + 1,5), fontsize = 14)
        plt.yticks(np.arange(-elevationRange, elevationRange + 1,5), fontsize = 14)
        plt.xlim((-azimuthRange, elevationRange))
        plt.ylim((-azimuthRange, elevationRange))
        plt.grid(True)

        #plt.savefig(dataPath + '/offline_data/gaze_accuracy_'+str(start_seconds)+'_'+ str(end_seconds)+'.png', dpi = 200 )
        plt.show()
    
    # Todo: This should not be here, most probably goes into vm_tools
    def rigid_transform_3D(A, B):
        assert len(A) == len(B)

        num_rows, num_cols = A.shape;

        if num_rows != 3:
            raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

        [num_rows, num_cols] = B.shape;
        if num_rows != 3:
            raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)
        print(centroid_A, np.tile(centroid_A, (num_cols,1)).T)

        # subtract mean
        Am = A - np.tile(centroid_A, (num_cols,1)).T
        Bm = B - np.tile(centroid_B, (num_cols,1)).T

        H = Am.dot(np.transpose(Bm))

        # sanity check
        #if linalg.matrix_rank(H) < 3:
        #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T.dot(U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            print("det(R) < R, reflection detected!, correcting for it ...\n");
            Vt[2,:] *= -1
            R = Vt.dot(U.T)

        t = -R.dot(centroid_A) + centroid_B

        return R, t


    # Todo: This should not be here, most probably goes into vm_tools
    def gaze_calibration_assessment(self, folder):
        import pandas as pd
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        import cv2
        from matplotlib import cm
        print("\n\nCalibration Assessment!")
        dataPath = '/hdd01/data_base_local/341/'
        print('dataPath [dummy]: ', dataPath)
        # TODO: For now I'm using a dummy folder
        # dataPath = folder
        # print('dataPath: ', dataPath)


        listOfImages = []
        markerFrames = []


        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #images = glob.glob('*.jpg')

        cap = cv2.VideoCapture(dataPath + 'world.mp4')
        numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( 'Total Number of Frames: ', numberOfFrames )
        #count = 1800
        #while(cap.isOpened()):

        fps = 30
        safeMargin = 2
        start_seconds = 70
        end_seconds = 120

        startIndex = (start_seconds + safeMargin) * fps
        endIndex = (end_seconds - safeMargin) * fps
        print('First Frame = %d'%(startIndex))
        print('Last Frame = %d'%(endIndex))

        scale_x = 0.5
        scale_y = 0.5
        print('scale[x,y] = ', scale_x, scale_y)

        myString = '-'
        for count in range(0, numberOfFrames):
            
            print("Progress: {0:.1f}% {s}".format(count*100/numberOfFrames, s = myString), end="\r", flush=True)
            if count < startIndex:
                ret, frame = cap.read()
                continue
            elif count > endIndex:
                break
            else:


                #Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
                ret, img = cap.read()

                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                gray = cv2.resize(gray,None,fx=scale_x,fy=scale_y)
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (6,8),None)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    #print('====> Found [%d]!' %(count))
                    objpoints.append(objp)
                    

                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)


                    # Draw and display the corners
                    img_1 = cv2.drawChessboardCorners(cv2.resize(img,None,fx=scale_x,fy=scale_y), (6,8), corners2,ret)
                    cv2.imshow('Frame',img_1)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break

                    corners2 = np.squeeze(corners2)
                    #print('before : ', corners)
                    corners2[:,0] = corners2[:,0]*(1/scale_x)
                    corners2[:,1] = corners2[:,1]*(1/scale_y)
                    #print('after : ', corners)

                    imgpoints.append(corners2)
                    listOfImages.append(img_1)
                    markerFrames.append(count)
                    myString = '1'
                else:
                    myString = '0'
                    #cv2.imshow('img',gray)
                    #if cv2.waitKey(5) & 0xFF == ord('q'):
                    #    break
        print('\nDone!')
        cv2.destroyAllWindows()
        corners_pixels = np.array(imgpoints)
        corners_pixels.shape
        markerPosition = np.mean(corners_pixels, axis = 1)
        markerPosition.shape
        #np.save(dataPath + '/offline_data/markerPosition_'+str(start_seconds)+'_'+ str(end_seconds)+'.npy', markerPosition)
        #np.save(dataPath + '/offline_data/corners_pixel_'+str(start_seconds)+'_'+ str(end_seconds)+'.npy', corners_pixels)

        worldTimeStamps = np.load(dataPath + 'world_timestamps.npy')
        gazeDataFrame = pd.read_csv(dataPath + 'exports/000/gaze_positions.csv')

        gazeIndex = []
        for markerIndex in markerFrames:
            i = np.argmin(np.abs((gazeDataFrame.gaze_timestamp.values - worldTimeStamps[markerIndex]).astype(float)))
            gazeIndex.append(i)
        self.plot_gaze_accuracy(markerPosition, gazeDataFrame, gazeIndex)
        self.plot_fps('world')
        self.plot_fps('eye0')


class RecordedData(Session):
    """ Base class for Data Object. """

    def __init__(self, session_id, main_directory, move = False,
                    export_directory = '/000/', data_type = 'world'):
        """ Constructor.

        Parameters
        ----------
        parameter_1: str
            Parameter_1 .

        """
        #self.main_directory = main_directory
        super().__init__(session_id, main_directory, move = False, export_directory = '/000/')
        self.data_type = data_type
        self.video = None
        self.video_file = os.path.join(self.main_directory + self.session_id, data_type +'.mp4')
        print('video file: ', self.video_file)

        self.time_stamp = None
        self.time_stamp_file = os.path.join(self.main_directory + self.session_id, data_type +'_timestamps.npy')
        print('timestamp file: ', self.time_stamp_file)
