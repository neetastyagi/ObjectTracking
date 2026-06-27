"""Problem Set 7: Particle Filter Tracking."""

import cv2
import numpy as np

from ps5_utils import run_kalman_filter, run_particle_filter

np.random.seed(42)  #DO NOT CHANGE THIS SEED VALUE

# I/O directories
input_dir = "input"
output_dir = "output"

# TODO: Remove unnecessary classes

# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.covariance = np.array([[10, 0, 0, 0],
                                    [0, 10, 0, 0],
                                    [0, 0, 10, 0],
                                    [0, 0, 0, 10]])     # covariance

        self.D = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])   # dynamic transition matrix

        self.M = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])   # measurement matrix

        self.Q = Q  # process noise
        self.R = R  # measurement noise
 #       raise NotImplementedError

    def predict(self):

        self.state = np.dot(self.D, self.state)
        self.covariance = np.dot(np.dot(self.D, self.covariance), self.D.T) + self.Q
        #raise NotImplementedError

    def correct(self, meas_x, meas_y):
        K = np.dot(np.dot(self.covariance, self.M.T),
                np.linalg.inv(np.dot(np.dot(self.M, self.covariance), self.M.T) + self.R))
        
        Y = np.array([meas_x, meas_y])

        c_state = self.state + np.dot(K, (Y - np.dot(self.M, self.state)))
        c_covariance = np.dot((np.eye(4) - np.dot(K, self.M)), self.covariance)

        self.state = c_state
        self.covariance = c_covariance


        
        #raise NotImplementedError

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)#template
        
        
        self.frame = frame

        height = frame.shape[0]
        width  = frame.shape[1]


 #       h, w = frame.shape[0], frame.shape[1]

        

        tx = self.template_rect.get('x')
        ty = self.template_rect.get('y')
        tw = self.template_rect.get('w')
        th = self.template_rect.get('h')

        print('self.template_rect',tx,ty,tw,th)

        

        c_x = int(tx + tw/2)
        c_y = int(ty + th/2)

        rand_h = np.expand_dims(np.random.normal(c_x, self.sigma_dyn, self.num_particles), 1)
        rand_w = np.expand_dims(np.random.normal(c_y, self.sigma_dyn, self.num_particles), 1)
        self.particles = np.concatenate((rand_h, rand_w), axis=1)

        
 #       self.particles = np.array([[np.random.choice(height), np.random.choice(width)] for i in range(self.num_particles)])  # Initialize your particles array. Read the docstring.
#        self.weights = np.ones(self.num_particles) / self.num_particles
        self.weights = np.array([1./self.num_particles] * self.num_particles)
        # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.

        #raise NotImplementedError

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        mse = np.sum(np.subtract(template.astype(np.float) , frame_cutout.astype(np.float))**2)

 #       mse = np.sum(np.square(np.abs(template - frame_cutout)))

        h = template.shape[0]
        w = template.shape[1]

        if (h == 0):
            h = 1

        if (w == 0):
            w = 1
            
            
            

        mse = float(mse / (h * w))
        
 #       mse = mse / np.float(template.shape[0] * template.shape[1])

        similarity = np.exp(-mse / (2 * (self.sigma_exp)**2))

#        similarity = np.exp(-mse / (2 * np.square(self.sigma_exp)))
        
        return similarity
        #return NotImplementedError

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        
        r_num_particles = np.random.choice(self.num_particles, self.num_particles, replace = True, p=self.weights)
        
        self.particles = self.particles[r_num_particles]

        return self.particles
        

##        print('self.particles inital',self.particles)
##        new_resampled_particles = self.particles
##
##        r_num_particles = np.random.choice(self.num_particles, self.num_particles, p=self.get_weights())
##        
##        for i, n in enumerate(r_num_particles):
##            new_resampled_particles[i] = self.particles[n]
##        return new_resampled_particles #NotImplementedError
    

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        h, w = self.template.shape[:2]
        f_h, f_w = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        

        normalization = 0.

        self.particles = self.resample_particles()
 #       print('self.particles',self.particles)
        
        for i in range(self.num_particles):

            p = self.particles[i]
            print('p',p)

            c_w, c_h = p.astype(int)
            t_h, t_w = self.template.shape[:2]
            
            
            fx_from, fx_to  = c_w - (t_w//2), c_w + (t_w//2)
            fy_from, fy_to  = c_h - (t_h//2), c_h + (t_h//2)

            print('before',fx_from, fx_to, fy_from, fy_to)
            
            
            if fx_from<1:
                fx_from, fx_to = 1, 1 + fx_to-fx_from
                
            if fy_from<1:
                fy_from, fy_to = 1, 1 + fy_to-fy_from
                
            if fx_to > f_w-1:
                fx_from, fx_to = fx_from-(fx_to-f_w)-1, f_w-1
                
            if fy_to > f_h-1:
                fy_from, fy_to = fy_from-(fy_to-f_h)-1, f_h-1

            if(fx_to-fx_from == w-1 and fy_to-fy_from == h-1):
                fx_to = fx_to +1
                fy_to = fy_to +1

            print('after',fx_from, fx_to, fy_from, fy_to)
            
            frame_cutout = gray_frame[fy_from:fy_to, fx_from:fx_to]



            print('frame_cutout',frame_cutout.shape)

            similarity = self.get_error_metric(self.template, frame_cutout)
 
            self.weights[i] = similarity
            normalization += self.weights[i]



            
##            u = np.random.randint(-self.sigma_dyn, self.sigma_dyn)
##            v = np.random.randint(-self.sigma_dyn, self.sigma_dyn)
            u = np.random.randint(-self.sigma_dyn, self.sigma_dyn + 1)
            v = np.random.randint(-self.sigma_dyn, self.sigma_dyn + 1)

            if( 0 > self.particles[i,0] + u > f_h):
                self.weights[i] = 0.
            if(0 > self.particles[i,1] + v > f_w):
                self.weights[i] = 0.
            else:
                self.particles[i] = self.particles[i] + [u, v]                    
                    
                
        if normalization > 0:
 #           self.weights /= normalization
            self.weights /= np.sum(self.weights)

        return None


    def frame_cutout(self, frame, center):
        '''
        Cutout template frame
        '''
       
        h, w = self.template.shape[:2]
        f_h, f_w        = frame.shape[:2]
        c_w, c_h        = center.astype(int)
        t_h, t_w        = self.template.shape[:2]
        
        
        fx_from, fx_to  = c_w - (t_w//2), c_w + (t_w//2)
        fy_from, fy_to  = c_h - (t_h//2), c_h + (t_h//2)

        print('before',fx_from, fx_to, fy_from, fy_to)
        
        
        if fx_from<1:
            fx_from, fx_to = 1, 1 + fx_to-fx_from
            
        if fy_from<1:
            fy_from, fy_to = 1, 1 + fy_to-fy_from
            
        if fx_to > f_w-1:
            fx_from, fx_to = fx_from-(fx_to-f_w)-1, f_w-1
            
        if fy_to > f_h-1:
            fy_from, fy_to = fy_from-(fy_to-f_h)-1, f_h-1

        if(fx_to-fx_from == w-1 and fy_to-fy_from == h-1):
            fx_to = fx_to +1
            fy_to = fy_to +1

        print('after',fx_from, fx_to, fy_from, fy_to)
        
        return frame[fy_from:fy_to, fx_from:fx_to]





    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.

        h, w = self.template.shape[:2]

        x = int(x_weighted_mean - (w / 2))
        
        y = int(y_weighted_mean - (h / 2)) 


        cv2.rectangle(frame_in, (x, y), (x + w, y + h), (255, 0, 0))

        for px, py in self.particles.astype(np.int64):
            cv2.circle(frame_in, (px, py), 1, (0, 0, 255), -1)
        
        distance = self.particles - [x_weighted_mean, y_weighted_mean]
        distance = np.sqrt(np.sum(distance ** 2, axis=1))
        radius = np.sum(distance) / self.num_particles
       # print('radius', radius)
        center = (int(x_weighted_mean), int(y_weighted_mean))
        cv2.circle(frame_in, center, int(radius), (0, 255, 0))
        



class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        
        """
        h, w = self.template.shape[:2]
        f_h, f_w = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        
        best = np.array([])
        high_similarity = 0.
        normalization = 0.

        self.particles = self.resample_particles()
        # for each particle
        for i in range(self.num_particles):


            p = self.particles[i]
            print('p',p)
            c_w, c_h = p.astype(int)
            t_h, t_w = self.template.shape[:2]
            
            
            fx_from, fx_to  = c_w - (t_w//2), c_w + (t_w//2)
            fy_from, fy_to  = c_h - (t_h//2), c_h + (t_h//2)

            print('before',fx_from, fx_to, fy_from, fy_to)
            
            
            if fx_from<1:
                fx_from, fx_to = 1, 1 + fx_to-fx_from
                
            if fy_from<1:
                fy_from, fy_to = 1, 1 + fy_to-fy_from
                
            if fx_to > f_w-1:
                fx_from, fx_to = fx_from-(fx_to-f_w)-1, f_w-1
                
            if fy_to > f_h-1:
                fy_from, fy_to = fy_from-(fy_to-f_h)-1, f_h-1

            if(fx_to-fx_from == w-1 and fy_to-fy_from == h-1):
                fx_to = fx_to +1
                fy_to = fy_to +1

            print('after',fx_from, fx_to, fy_from, fy_to)
            
            frame_cutout = gray_frame[fy_from:fy_to, fx_from:fx_to]

            # calculate the similarity
            similarity = self.get_error_metric(self.template, frame_cutout) 

            
            if similarity > high_similarity:
                best = frame_cutout
                high_similarity = similarity
            


            u = np.random.randint(-self.sigma_dyn, self.sigma_dyn + 1)
            v = np.random.randint(-self.sigma_dyn, self.sigma_dyn + 1)

            if( 0 > self.particles[i,0] > f_h - 1):
                self.weights[i] = 0.
            if( 0 > self.particles[i,1] > f_w - 1):
                self.weights[i] = 0.
            else:
                self.weights[i] = similarity#self.alpha * similarity + (1 - self.alpha) * self.weights[i]
                normalization += self.weights[i]

            if(0 > self.particles[i,0]):
                self.particles[i] = self.particles[i] + [0., v]
            if(0 > self.particles[i,1]):
                self.particles[i] = self.particles[i] + [u , 0.]
            if(self.particles[i,0] > f_h):
                self.particles[i] = [f_h -1 , self.particles[i,1] ]
            if(self.particles[i,1] > f_w):
                self.particles[i] = [ self.particles[i,0] , f_w -1 ]
            else:
                self.particles[i] = self.particles[i] + [u, v] 

        self.template  = self.alpha * best + (1 - self.alpha) * self.template



        if normalization > 0:

            self.weights /= np.sum(self.weights)
        



class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.pos_change = (0, 0)
        self.mse_change = 0

        self.mse   = 50
        self.scale = .99


        
    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        a_h, a_w = self.template.shape[:2]
        a_f_h, a_f_w = frame.shape[:2]
        a_gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)


        similarity_ls = []
        template_list = []
        scale_list = []
        normalization = 0.

        for t, p in enumerate(self.particles):
            
            scale = np.random.randint(80, 100 + 1) / 100.
            template = self.template.copy()
            template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            template_list.append(template)
            scale_list.append(scale)

            p = self.particles[t]
            print('p',p)
            c_w, c_h = p.astype(int)
            t_h, t_w = template.shape[:2]
            
            
            fx_from, fx_to  = c_w - (t_w//2), c_w + (t_w//2)
            fy_from, fy_to  = c_h - (t_h//2), c_h + (t_h//2)

            print('before',fx_from, fx_to, fy_from, fy_to)
            
            
            if fx_from<1:
                fx_from, fx_to = 1, 1 + fx_to-fx_from
                
            if fy_from<1:
                fy_from, fy_to = 1, 1 + fy_to-fy_from
                
            if fx_to > a_f_h-1:
                fx_from, fx_to = fx_from-(fx_to-a_f_h)-1, a_f_h-1
                
            if fy_to > a_f_h-1:
                fy_from, fy_to = fy_from-(fy_to-a_f_h)-1, a_f_h-1

            if(fx_to-fx_from == a_w-1 and fy_to-fy_from == a_h-1):
                fx_to = fx_to +1
                fy_to = fy_to +1

            print('after',fx_from, fx_to, fy_from, fy_to)
            
            frame_cutout = a_gray_frame[fy_from:fy_to, fx_from:fx_to]

            resized_frame_cut = cv2.resize(src=frame_cutout, dsize=(0, 0), fx=self.scale, fy=self.scale).astype(np.uint8)

            if template.shape != resized_frame_cut.shape:
                resized_frame_cut = cv2.resize(src=resized_frame_cut, dsize=template.shape[::-1]).astype(np.uint8)

            similarity = self.get_error_metric(template, resized_frame_cut)

            similarity_ls.append(similarity)
            self.weights[t] += similarity
            normalization += self.weights[t]

        idx = np.argmax(similarity_ls)
        if scale_list[idx] > .95:
            self.template = template_list[idx]

        if normalization > 0:
            self.weights /= normalization
            self.weights /= np.sum(self.weights)
        
 #       raise NotImplementedError


def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    #500, 4, 20
    #500,20,12
    num_particles = 500  # Define the number of particles
    sigma_mse = 20#4  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 11#20  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    #500,5,18
    #200,6,15
    num_particles = 100  # Define the number of particles
    sigma_mse = 20  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 14  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    #500, 16, 13, 0.65
    num_particles = 500  # Define the number of particles
    sigma_mse = 4.5 #16  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 17.5 #13  # Define the value of sigma for the particles movement (dynamics)
    alpha = .58 # Set a value for alpha

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 200  # Define the number of particles
    sigma_md = 18  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to
    return out
