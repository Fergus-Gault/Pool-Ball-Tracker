import time
import logging
import numpy as np

from core import config, state

logger = logging.getLogger(__name__)

class StateManager():
    def __init__(self):
        self.previous_state = None
        self.time_since_last_update = time.time() - config.network_update_interval
        self.end_of_turn = False
        self.not_moved_counter = 0


    def update(self, detections, labels):
        if state.network and state.network.positions_requested:
            self.previous_state = None
            state.network.positions_requested = False

        current_time = time.time()

        if current_time - self.time_since_last_update < config.network_update_interval:
            return

        if not detections or not detections[0].boxes:
            self._handle_end_of_turn()
            return

        balls = {}
        corrected_white_ball = {}
        num_balls = 0
        self.not_moved_counter = 0

        x_ratio = np.divide(config.output_dimensions[0], (config.output_dimensions[0] - (2 * config.gantry_effective_range_x_px[0])))
        y_ratio = np.divide(config.output_dimensions[1], (config.output_dimensions[1] - (2 * config.gantry_effective_range_y_px[0])))

        for ball in detections[0].boxes:
            classname, middlex, middley = self._get_ball_info(ball, labels)
            if classname in {"arm", "hole"}:
                continue

            num_balls += 1

            if classname == "white":
                corrected_middlex, corrected_middley = self._handle_offset(
                    middlex, middley, x_ratio, y_ratio)
                
                corrected_white_ball.update({
                    "x": corrected_middlex, 
                    "y": corrected_middley}) 

            if self.previous_state and classname in self.previous_state:
                for prev_ball in self.previous_state[classname]:
                    if self._near_previous_position(prev_ball, middlex, middley):
                        self.not_moved_counter += 1
                        prev_ball["x"] = middlex
                        prev_ball["y"] = middley
                        break

            balls.setdefault(classname, []).append(
                {"x": middlex, 
                 "y": middley})

        if self.not_moved_counter == num_balls:
            self.previous_state = balls
            self._handle_end_of_turn()
            return

        self._update_and_send_balls(balls, corrected_white_ball, current_time)



    def _get_ball_info(self, ball, labels):
        xyxy_tensor = ball.xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = map(int, xyxy.astype(int))
        classidx : int = int(ball.cls.item())
        classname : str = labels[classidx]
        _middlex : int = int((xmin + xmax) // 2)
        _middley : int = int((ymin + ymax) // 2)

        middlex, middley = self._coords_clamped(_middlex, _middley)
        return classname, middlex, middley
    

    def _coords_clamped(self, x, y):
        x = max(0, min(x, config.output_dimensions[0]))
        y = max(0, min(y, config.output_dimensions[1]))
        return int(x), int(y)
    
    
    def _near_previous_position(self, prev_ball, x, y):
        return abs(prev_ball["x"] - x) < config.position_threshold \
            and abs(prev_ball["y"] - y) < config.position_threshold
    

    def _handle_end_of_turn(self):
        if self.end_of_turn:
            return
        self.end_of_turn = True
        if config.use_networking and state.network:
            state.network.send_end_of_turn("true")
        else:
            logger.info("No movement detected, end of turn.")


    def _update_and_send_balls(self, balls, corrected_white_ball, current_time):
        if not balls:
            return

        self.previous_state = balls
        self.time_since_last_update = current_time
        self.end_of_turn = False

        if config.use_networking and state.network:
            state.network.send_balls({"balls": balls})
            if corrected_white_ball:
                state.network.send_corrected_white_ball(corrected_white_ball)
        else:
            logger.info(f"Sending balls: {balls}")
            if corrected_white_ball:
                logger.info(
                    f"Sending corrected white ball: {corrected_white_ball}")    
                

    def _handle_offset(self, middlex, middley, x_ratio, y_ratio):
        corrected_middlex = self._handle_x_offset(middlex, x_ratio)
        corrected_middley = self._handle_y_offset(middley, y_ratio)
        corrected_middlex, corrected_middley = self._coords_clamped(
            corrected_middlex, corrected_middley)
        return corrected_middlex, corrected_middley

    def _handle_x_offset(self, middlex, x_ratio):
        if middlex > config.gantry_effective_range_x_px[1]:
            return config.output_dimensions[0]
        elif middlex < config.gantry_effective_range_x_px[0]:
            return 0
        return (middlex - config.gantry_effective_range_x_px[0]) * x_ratio
    
    def _handle_y_offset(self, middley, y_ratio):
        if middley > config.gantry_effective_range_y_px[1]:
            return config.output_dimensions[1]
        elif middley < config.gantry_effective_range_y_px[0]:
            return 0
        return (middley - config.gantry_effective_range_y_px[0]) * y_ratio
