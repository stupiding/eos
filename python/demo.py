import eos
import dlib, cv2
import numpy as np

import os, time

class TestEOS():
    def __init__(self):
        """Demo for running the eos fitting from Python."""
        self.landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings

        self.model = eos.morphablemodel.load_model("../data/sfm_shape_3448.bin")
        self.blendshapes = eos.morphablemodel.load_blendshapes("../data/expression_blendshapes_3448.bin")
        self.landmark_mapper = eos.core.LandmarkMapper('../data/ibug_to_sfm.txt')
        self.edge_topology = eos.morphablemodel.load_edge_topology('../data/sfm_3448_edge_topology.json')
        self.contour_landmarks = eos.fitting.ContourLandmarks.load('../data/ibug_to_sfm.txt')
        self.model_contour = eos.fitting.ModelContour.load('../data/model_contours.json')

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('../data/shape_predictor_68_face_landmarks.dat')

    def get_3x4_affine_camera_matrix(self,  params, width, height):
        ortho_projection = params.get_projection()
        modelview = params.get_modelview()
        mvp = np.dot(ortho_projection, modelview)

        viewport = np.array([0, height, width, -height])
        
        viewport_matrix = np.zeros((4, 4))
        viewport_matrix[0, 0] = 0.5 * viewport[2]
        viewport_matrix[0, 3] = 0.5 * viewport[2] + viewport[0]
        viewport_matrix[1, 1] = 0.5 * viewport[3]
        viewport_matrix[1, 3] = 0.5 * viewport[3] + viewport[1]
        viewport_matrix[2, 2] = 1
        viewport_matrix[3, 3] = 1

        full_projection_4x4 = np.dot(viewport_matrix, mvp)
        full_projection_3x4 = full_projection_4x4[:3, :]
        full_projection_3x4[2, :] = np.array([0, 0, 0, 1])

        return full_projection_3x4

    def calculate_affine_z_direction(self, full_projection_3x4):
        affine_cam_z_rotation = np.cross(full_projection_3x4[0, :3], full_projection_3x4[1, :3])
        affine_cam_z_rotation /= np.linalg.norm(affine_cam_z_rotation)

        affine_cam_4x4 = np.zeros([4, 4]);
        affine_cam_4x4[:2, :] = full_projection_3x4[:2, :]
        affine_cam_4x4[2, :3] = affine_cam_z_rotation
        affine_cam_4x4[3, 3] = 1

        return affine_cam_4x4

    def test(self, image_name):
        # Or for example extract the texture map, like this:
        image = cv2.imread(image_name)
        image_height, image_width = image.shape[:2]

        start_time = time.time()
        landmarks, rects = self.get_landmark(image)
        landmark_time = time.time()
        landmarks = landmarks[0]

        (mesh, params, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(self.model, self.blendshapes,
            landmarks, self.landmark_ids, self.landmark_mapper,
            image_width, image_height, self.edge_topology, self.contour_landmarks, self.model_contour, 1)
        fitting_time = time.time()

        full_projection_3x4 = self.get_3x4_affine_camera_matrix(params, image_width, image_height)
        affine_cam_4x4 = self.calculate_affine_z_direction(full_projection_3x4)

        v = np.asarray(mesh.vertices)
        v[:, -1] = 1.0
        points_2d = np.dot(np.copy(v), affine_cam_4x4.T)

        image_cp = image.copy()
        for point in landmarks:
            cv2.circle(image_cp, (int(point[0]), int(point[1])), 5, (255,0,0), -1)
        
        for i, point in enumerate(points_2d):
            cv2.circle(image_cp, (int(point[0]), int(point[1])), 1, (0,0,255), -1)
        cv2.imshow('res.png', image_cp)

        depth_im = eos.render.render_affine(mesh, params, image_width, image_height, do_backface_culling=True)
        max_depth = depth_im[depth_im<depth_im.max()].max()
        depth_im[depth_im > max_depth] = 100
        depth_im /= depth_im.max()
        cv2.imshow('depth', depth_im)

        isomap = eos.render.extract_texture(mesh, params, image)
        render_time = time.time()

        print("landmark time: %f, fitting_time: %f, render_time: %f" % (landmark_time - start_time, fitting_time - landmark_time, render_time - fitting_time))
        cv2.imshow("a", isomap)
        cv2.waitKey(0)
       
    def get_landmark(self, image):
        imgScale = 1
        dets = self.detector(image, 1)
        shapes2D = []
        for det in dets:
            faceRectangle = dlib.rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale), int(det.bottom() / imgScale))

            dlibShape = self.predictor(image, faceRectangle)
            shape2D = [[p.x, p.y] for p in dlibShape.parts()]

            shapes2D.append(shape2D)
        return (shapes2D, dets)

if __name__ == "__main__":
    test_eos = TestEOS()
    root_folder = '../examples/data/female/'
    files = os.listdir(root_folder)
    for img in files:
        test_eos.test(root_folder + img)
