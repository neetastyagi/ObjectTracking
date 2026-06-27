"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import os

import cv2
import numpy as np

import ps5

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


def part_1b():
    print("Part 1b")

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-b-1.png'),
        28: os.path.join(output_dir, 'ps5-1-b-2.png'),
        57: os.path.join(output_dir, 'ps5-1-b-3.png'),
        97: os.path.join(output_dir, 'ps5-1-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1b(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "circle"))


def part_1c():
    print("Part 1c")

    template_loc = {'x': 311, 'y': 217}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-c-1.png'),
        30: os.path.join(output_dir, 'ps5-1-c-2.png'),
        81: os.path.join(output_dir, 'ps5-1-c-3.png'),
        155: os.path.join(output_dir, 'ps5-1-c-4.png')
    }

    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1c(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "walking"))


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {
        8: os.path.join(output_dir, 'ps5-2-a-1.png'),
        28: os.path.join(output_dir, 'ps5-2-a-2.png'),
        57: os.path.join(output_dir, 'ps5-2-a-3.png'),
        97: os.path.join(output_dir, 'ps5-2-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2a(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "circle"))


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {
        12: os.path.join(output_dir, 'ps5-2-b-1.png'),
        28: os.path.join(output_dir, 'ps5-2-b-2.png'),
        57: os.path.join(output_dir, 'ps5-2-b-3.png'),
        97: os.path.join(output_dir, 'ps5-2-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2b(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "pres_debate_noisy"))


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {
        20: os.path.join(output_dir, 'ps5-3-a-1.png'),
        48: os.path.join(output_dir, 'ps5-3-a-2.png'),
        158: os.path.join(output_dir, 'ps5-3-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_3(
        ps5.AppearanceModelPF,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pres_debate"))


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {
        40: os.path.join(output_dir, 'ps5-4-a-1.png'),
        100: os.path.join(output_dir, 'ps5-4-a-2.png'),
        240: os.path.join(output_dir, 'ps5-4-a-3.png'),
        300: os.path.join(output_dir, 'ps5-4-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_4(
        ps5.MDParticleFilter,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pedestrians"))


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    template_rect1 = {'x': 60, 'y': 150, 'w': 80, 'h': 130}
    template_rect2 = {'x': 270, 'y': 200, 'w': 100, 'h': 130}
    template_rect3 = {'x': 0, 'y': 160, 'w': 54, 'h': 130}
    num_particles1 = 600
    sigma_exp1 = 3
    sigma_dyn1 = 30
    num_particles2 = 600
    sigma_exp2 = 3
    sigma_dyn2 = 30
    num_particles3 = 600
    sigma_exp3 = 3
    sigma_dyn3 = 30
    alpha = 0.55
    template_coords1 = template_rect1
    template_coords2 = template_rect2
    template_coords3 = template_rect3
    save_frames = {
        29: os.path.join(output_dir, 'ps5-5-a-1.png'),
        56: os.path.join(output_dir, 'ps5-5-a-2.png'),
        71: os.path.join(output_dir, 'ps5-5-a-3.png')
    }
    filter_class1 = ps5.ParticleFilter
    filter_class2 = ps5.ParticleFilter
    imgs_dir = os.path.join(input_dir, "TUD-Campus")
    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    template1 = None
    template2 = None
    template3 = None
    pf1 = None
    pf2 = None
    pf3 = None
    frame_num = 1

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))
        if frame_num < 64:
            if template1 is None:
                template1 = frame[int(template_rect1['y']):
                                  int(template_rect1['y'] + template_rect1['h']),
                            int(template_rect1['x']):
                            int(template_rect1['x'] + template_rect1['w'])]


                pf1 = filter_class1(frame, template1, num_particles=num_particles1, sigma_exp=sigma_exp1,
                                    sigma_dyn=sigma_dyn1,
                                    template_coords=template_coords1, alpha=alpha)

            pf1.process(frame)

            if True:
                out_frame = frame.copy()
                pf1.render(out_frame)

        if frame_num < 49:
            if template2 is None:
                frame2 = out_frame.copy()

                template2 = frame2[int(template_rect2['y']):
                                   int(template_rect2['y'] + template_rect2['h']),
                            int(template_rect2['x']):
                            int(template_rect2['x'] + template_rect2['w'])]


                pf2 = filter_class2(frame2, template2, num_particles=num_particles2, sigma_exp=sigma_exp2,
                                    sigma_dyn=sigma_dyn2, template_coords=template_coords2, alpha=alpha)
            else:
                frame2 = out_frame.copy()
            pf2.process(frame2)

            if frame_num < 24:
                if True:
                    out_frame2 = frame2.copy()
                    pf2.render(out_frame2)
                    cv2.imshow('Tracking', out_frame2)
                    cv2.waitKey(1)

                if frame_num in save_frames:
                    frame_out = frame2.copy()
                    pf2.render(frame_out)
                    cv2.imwrite(save_frames[frame_num], frame_out)
            else:
                if True:
                    out_frame2 = frame2.copy()
                    pf2.render(out_frame2)

        if frame_num > 23:
            print('frame num', frame_num)
            if template3 is None:
                frame3 = out_frame2.copy()
                template3 = frame3[int(template_rect3['y']):
                                   int(template_rect3['y'] + template_rect3['h']),
                            int(template_rect3['x']):
                            int(template_rect3['x'] + template_rect3['w'])]
##                cv2.imshow('template3', template3)
##                cv2.waitKey(1)

                pf3 = filter_class1(frame3, template3, num_particles=num_particles3, sigma_exp=sigma_exp3,
                                    sigma_dyn=sigma_dyn3,
                                    template_coords=template_coords3, alpha=alpha)
            else:
                if frame_num > 63:
                    frame3 = frame.copy()
                elif frame_num > 48:
                    frame3 = out_frame.copy()
                else:
                    frame3 = out_frame2.copy()
            pf3.process(frame3)

            if True:
                out_frame3 = frame3.copy()
                pf3.render(out_frame3)
                cv2.imshow('Tracking', out_frame3)
                cv2.waitKey(1)
        if frame_num in save_frames:
            frame_out = frame3.copy()
            pf3.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))
    return 0


def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    img_dir = os.path.join(input_dir, "follow")
    img_list = [f for f in os.listdir(img_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    img_list.sort()


    template_rect = {'x': 90, 'y': 30, 'w': 44, 'h': 150}

    save_frames = {
        69: os.path.join(output_dir, 'ps5-6-a-1.png'),
        160: os.path.join(output_dir, 'ps5-6-a-2.png'),
        186: os.path.join(output_dir, 'ps5-6-a-3.png'),
    }


    template = None
    pf = None
    frame_n = 1
    f_c = ps5.AppearanceModelPF
    
    for img in img_list:

        

        fr = cv2.imread(os.path.join(img_dir, img))
        if template is None:
            template = fr[int(template_rect['y']):
                                 int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):
                                 int(template_rect['x'] + template_rect['w'])]


#            pf = ps5.AppearanceModelPF(frame, template, num_particles=500, sigma_exp=3, sigma_dyn=14, alpha = 0.55, template_coords=template_rect)
            #pf = f_c(fr, template, num_particles=500, sigma_exp=3, sigma_dyn=10.5, alpha = 0.60, template_coords=template_rect)
            pf = f_c(fr, template, num_particles=500, sigma_exp=3, sigma_dyn=9, alpha=0.55,
                     template_coords=template_rect)
        pf.process(fr)

        if True:  # For debugging, it displays every frame
            out_frame = fr.copy()
            pf.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_n in save_frames:
            frame_out = fr.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_n], frame_out)

        # Update frame number
        frame_n += 1
        if frame_n % 20 == 0:
            print('Working on frame {}'.format(frame_n))


if __name__ == '__main__':
    part_1b()
    part_1c()
    part_2a()
    part_2b()
    part_3()
    part_4()
    part_5()
    part_6()
