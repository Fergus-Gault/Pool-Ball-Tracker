from liveconfig import LiveConfig, start_interface
LiveConfig("./src/data")
import logging
import cv2
import numpy as np
from core import config, state, state_manager, load_camera, parse_args, capture_frame
from src.processing.camera_processing import get_top_down_view, handle_calibration, undistort_frame, manage_point_selection
from src.detection.detection import DetectionModel
import time

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    handlers=[
        logging.StreamHandler()
    ]
)

def main():

    prev_frame_time = 0 

    new_frame_time = 0
    args = parse_args()
    if not args.no_interface:
        start_interface("web")

    if args.file is not None:
        frame = cv2.imread(args.file)
    else:
        camera = load_camera()
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to read from camera.")
            return
        
    processed_frame = frame
    mtx, dist, newcameramtx, roi = handle_calibration(frame)
    processed_frame = undistort_frame(frame, mtx, dist, newcameramtx, roi)

    table_pts = manage_point_selection(processed_frame)
    if table_pts is None:
        logger.error("Table points not selected. Continuing without.")
        config.use_table_pts = False
    else:
        table_rect = np.float32([
            [0, 0],
            [config.output_dimensions[0], 0],
            [0, config.output_dimensions[1]],
            [config.output_dimensions[0], config.output_dimensions[1]]
        ])
        homography_matrix = cv2.getPerspectiveTransform(table_pts, table_rect)
        processed_frame = get_top_down_view(processed_frame, homography_matrix)
        
    detection_model = DetectionModel()
    if detection_model.model is None:
        return
    
    while True:

        if args.file is None:
            ret, frame = camera.read()
            if frame is None:
                logger.error("Failed to read from camera.")
        
        processed_frame = frame

        if config.use_calibration:
            processed_frame = undistort_frame(frame, mtx, dist, newcameramtx, roi)
        if config.use_table_pts:
            processed_frame = get_top_down_view(processed_frame, homography_matrix)
        if config.collect_model_images or config.collect_ae_data:
            capture_frame(processed_frame)

        detections, labels = detection_model.handle_detection(processed_frame)
        state_manager.update(detections, labels)

        if state.autoencoder is not None \
            and config.use_obstruction_detection \
            and config.use_model:
            table_only = detection_model.extract_bounding_boxes(
                processed_frame, 
                detections)
            state.autoencoder.handle_obstruction_detection(table_only)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        cv2.putText(processed_frame, 
                    f"FPS: {fps}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    config.font_scale, 
                    config.font_color, 
                    config.font_thickness)
        
        cv2.imshow("Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        
    if args.file is None:
        camera.release()
    cv2.destroyAllWindows()
    if config.use_networking:
        state.network.disconnect()
    
    
if __name__ == "__main__":
    main()