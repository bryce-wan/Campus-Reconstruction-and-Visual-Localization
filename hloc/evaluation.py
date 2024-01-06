import pycolmap
import time
import numpy as np
import time
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.visualization import plot_images, read_image
from hloc.utils.read_write_model import read_images_binary, qvec2rotmat
from hloc.utils.viz import plot_images, plot_keypoints, save_plot
from hloc.utils import viz_3d


class evaluation():
    def __init__(self, work_dir, images, features, matches):
        self.outputs = work_dir
        self.model = pycolmap.Reconstruction(self.outputs / 'sfm')
        print(self.model.summary())
        self.features = features
        self.matches = matches
        self.images = images
        self.references = []

        ids = []
        for i in self.model.images.items():
            ids.append(i[0])
        for i in ids:
            self.references.append(self.model.images[i].name)

    def model_viz_3D(self):
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, self.model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True, cameras=True)
        fig.show()

    def kps_viz_3D(self, query, ret, log, camera):
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, self.model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)

        pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
        viz_3d.plot_camera_colmap(
            fig, pose, camera, color='rgba(0,255,0,0.5)', name=query, fill=True)
        # visualize 2D-3D correspodences
        inl_3d = np.array([self.model.points3D[pid].xyz for pid in np.array(
            log['points3D_ids'])[ret['inliers']]])
        print(len(inl_3d))
        viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)
        fig.show()

    def calc_reprojection_error(self, query, max_error=1, viz=False, dpi=400):
        references = self.references
        model = self.model
        images = self.images
        features = self.features
        matches = self.matches

        rets = []
        logs = []
        cameras = []

        if not isinstance(query, list):
            query = [query]

        query.sort()
        for i in range(len(query)):
            q = query[i]
            print(f'Localizing {q}...')

            camera = pycolmap.infer_camera_from_image(images / q)
            cameras.append(camera)
            ref_ids = [model.find_image_with_name(
                r).image_id for r in references]
            conf = {
                'estimation': {'ransac': {'max_error': max_error}},
                'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
            }
            localizer = QueryLocalizer(model, conf)

            # calculate the time cost
            timer = time.time()
            ret, log = pose_from_cluster(
                localizer, q, camera, ref_ids, features, matches)
            rets.append(ret)
            logs.append(log)
            print(f'        Localization took {time.time() - timer:.3f}s.')
            print(
                f'        Found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')

            # plot the results
            inliers = np.array(log['PnP_ret']['inliers'])
            p2_pos = log["keypoints_query"][inliers]
            p3_ids = np.array(log["points3D_ids"])[inliers]
            p3_ids = p3_ids.tolist()
            reprojection_p2_pos = np.array([camera.world_to_image(pycolmap.Image(tvec=ret['tvec'],
                                                                                 qvec=ret['qvec']).project(model.points3D[id].xyz))for id in p3_ids])

            reprojection_p2_pos = np.around(
                reprojection_p2_pos, 1).astype(float)
            mean_reprojection_error = np.mean(
                np.linalg.norm(p2_pos - reprojection_p2_pos, axis=1))
            print(
                f"        The mean reprojection error is : {mean_reprojection_error}")

            if viz:
                q_image = read_image(images / q)
                results = self.outputs / 'result_images/'
                results.mkdir(parents=True, exist_ok=True)
                plot_images([q_image], dpi=dpi)
                plot_keypoints([p2_pos], colors='red', ps=6)
                plot_keypoints([reprojection_p2_pos], colors='green', ps=6)
                filename = 'query_' + \
                    '{:0>3d}'.format(i+1) + '_reprojection.png'
                save_plot(results / filename)

        return rets, logs, cameras
