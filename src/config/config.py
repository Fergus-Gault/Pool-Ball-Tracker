from liveconfig import liveclass

@liveclass
class Config:
    def __init__(self):
        self.font_color = (255, 255, 255)
        self.font_scale = 0.5
        self.font_thickness = 1
        self.bbox_colors = [
            (255,0,0), 
            (0,0,0), 
            (0, 165, 255), 
            (0,0,255), 
            (255,255,255), 
            (0,255,255)]
        self.use_obstruction_detection = True
        self.autoencoder_model_path = "./model/autoencoder_model.keras"
        self.collect_ae_data = False
        self.ae_data_path = "./src/data/model/ae_data/"
        self.obstruction_threshold = 0.013
        self.obstruction_warn_if_within = 0.05
        self.obstruction_buffer_size = 4
        self.use_networking = True
        self.network_update_interval = 0.1
        self.poolpal_url = "http://poolpal.joshn.uk"
        self.output_dimensions = (1200, 600)
        self.ball_area_range = (1500, 4500)
        self.arm_area_range = (10000, 20000)
        self.use_hidden_balls = False
        self.use_model = True
        self.process_every_n_frames = 2
        self.detection_model_path = "./model/detection_model.pt"
        self.position_threshold = 6
        self.hole_threshold = 30
        self.conf_threshold = 0.5
        self.draw_results = True
        self.hide_windows = False
        self.use_calibration = True
        self.use_table_pts = True
        self.model_image_path = "./src/data/model/training_images/"
        self.collect_model_images = False
        self.calibration_params_path = "./src/data/calibration_params.json"
        self.calibration_images_path = "./src/data/calibration_images/wide_angle_cam/"
        self.table_pts_path = "./src/data/table_pts.json"
