import cv2
import numpy as np
import pytesseract
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple


@dataclass
class ClassSession:
    course_code: str
    session_type: str
    building: str
    room: str
    day: str
    start_time: str
    end_time: str
    days: str = ""
    raw_text: str = ""
    confidence: float = 1.0


class ScheduleParser:
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    DAY_LETTER = {"Monday": "M", "Tuesday": "T", "Wednesday": "W", "Thursday": "R", "Friday": "F"}

    START_HOUR = 9

    def __init__(self, image_bytes: bytes, debug: bool = False):
        self.debug = debug

        np_img = np.frombuffer(image_bytes, np.uint8)
        self.image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError("Invalid image input")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.classes: List[ClassSession] = []
        self.header_bottom = 0
        self.time_col_right = 0

    def detect_header_region(self) -> int:
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        rows_with_blue = np.any(mask > 0, axis=1)
        if np.any(rows_with_blue):
            return np.where(rows_with_blue)[0][-1] + 5

        return int(self.image.shape[0] * 0.1)

    def detect_time_column(self) -> int:
        edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        detect_vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        vertical_projection = np.sum(detect_vertical, axis=0)
        threshold = np.max(vertical_projection) * 0.3
        candidates = np.where(vertical_projection > threshold)[0]
        candidates = candidates[candidates > 50]

        if len(candidates) > 0:
            return candidates[0]

        return int(self.image.shape[1] * 0.12)

    def detect_grid_lines(self) -> Tuple[List[int], List[int]]:
        thresh = cv2.adaptiveThreshold(
            self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        horizontal_projection = np.sum(detect_horizontal, axis=1)
        threshold_h = np.max(horizontal_projection) * 0.4

        potential_lines = []
        for y in range(len(horizontal_projection)):
            if horizontal_projection[y] > threshold_h and y > self.header_bottom:
                if not potential_lines or (y - potential_lines[-1]) > 5:
                    potential_lines.append(y)

        h_lines = []
        if potential_lines:
            h_lines.append(potential_lines[0])
            for line in potential_lines[1:]:
                if line - h_lines[-1] > 80:
                    h_lines.append(line)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        vertical_projection = np.sum(detect_vertical, axis=0)
        threshold_v = np.max(vertical_projection) * 0.4

        v_lines = []
        for x in range(len(vertical_projection)):
            if vertical_projection[x] > threshold_v and x > self.time_col_right:
                if not v_lines or (x - v_lines[-1]) > 20:
                    v_lines.append(x)

        return h_lines, v_lines

    def detect_class_boxes(self, v_lines: List[int]) -> List[dict]:
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        lower_sat = np.array([0, 5, 40])
        upper_sat = np.array([180, 255, 255])
        mask_sat = cv2.inRange(hsv, lower_sat, upper_sat)

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, mask_bright = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY_INV)

        mask = cv2.bitwise_or(mask_sat, mask_bright)

        mask[0:self.header_bottom, :] = 0
        mask[:, 0:self.time_col_right] = 0

        kernel_close = np.ones((7, 7), np.uint8)
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        for v_line in v_lines:
            cv2.line(mask, (v_line - 2, 0), (v_line - 2, mask.shape[0]), 0, 5)
            cv2.line(mask, (v_line, 0), (v_line, mask.shape[0]), 0, 5)
            cv2.line(mask, (v_line + 2, 0), (v_line + 2, mask.shape[0]), 0, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 30:
                boxes.append({"x": x, "y": y, "width": w, "height": h})

        return boxes

    def extract_text_from_box(self, box: dict) -> str:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        roi = self.image[y : y + h, x : x + w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.convertScaleAbs(roi_gray, alpha=1.5, beta=0)
        _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(roi_thresh, config="--psm 6").strip()

    def parse_class_text(self, text: str) -> dict:
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        result = {"course_code": "", "session_type": "", "building": "", "room": ""}

        if not lines:
            return result

        result["course_code"] = lines[0]
        if len(lines) >= 2:
            result["session_type"] = lines[1]

        if len(lines) >= 3:
            location_text = " ".join(lines[2:])
            tokens = location_text.split()
            if tokens:
                result["room"] = tokens[-1]
                result["building"] = " ".join(tokens[:-1])

        return result

    def map_box_to_time_day(self, box: dict, h_lines: List[int], v_lines: List[int]) -> Tuple[str, str, str]:
        box_center_x = box["x"] + box["width"] // 2
        box_top_y = box["y"]
        box_bottom_y = box["y"] + box["height"]

        if len(v_lines) >= 2:
            for i in range(len(v_lines) - 1):
                if v_lines[i] <= box_center_x < v_lines[i + 1]:
                    day = self.DAYS[i] if i < len(self.DAYS) else "Unknown"
                    break
            else:
                day = self.DAYS[-1]
        else:
            schedule_left = self.time_col_right
            schedule_width = self.image.shape[1] - schedule_left
            day_width = schedule_width / len(self.DAYS)
            day_idx = int((box_center_x - schedule_left) / day_width)
            day = self.DAYS[day_idx] if 0 <= day_idx < len(self.DAYS) else "Unknown"

        if len(h_lines) >= 2:
            first_line = h_lines[0]
            last_line = h_lines[-1]
            num_hours = len(h_lines) - 1
            total_pixels = last_line - first_line
            avg_pixels_per_hour = total_pixels / num_hours if num_hours > 0 else 100

            start_hour = self.START_HOUR + (box_top_y - first_line) / avg_pixels_per_hour
            end_hour = self.START_HOUR + (box_bottom_y - first_line) / avg_pixels_per_hour

            start_time = self.format_time(start_hour, round_to_5=True)
            end_time = self.format_time(end_hour, round_to_5=True)
        else:
            schedule_top = self.header_bottom
            schedule_height = self.image.shape[0] - schedule_top
            pixels_per_hour = schedule_height / 8
            start_time = self.format_time(self.START_HOUR + (box_top_y - schedule_top) / pixels_per_hour, round_to_5=True)
            end_time = self.format_time(self.START_HOUR + (box_bottom_y - schedule_top) / pixels_per_hour, round_to_5=True)

        return day, start_time, end_time

    def format_time(self, hour: float, round_to_5: bool = False) -> str:
        h = int(hour)
        m = int((hour - h) * 60)
        if round_to_5:
            m = round(m / 5) * 5
            if m == 60:
                h += 1
                m = 0

        period = "AM" if h < 12 else "PM"
        display_h = h if h <= 12 else h - 12
        if display_h == 0:
            display_h = 12
        return f"{display_h}:{m:02d} {period}"

    def merge_rows(self, rows):
        merged = {}
        for r in rows:
            key = (
                r["course_code"].replace(" ", "").upper(),
                r["session_type"].replace(" ", "").upper(),
                r["building"],
                r["room"],
                r["start_time"],
                r["end_time"],
            )
            day_letter = self.DAY_LETTER.get(r["day"], "")

            if key not in merged:
                merged[key] = {**r, "days": day_letter}
            else:
                merged[key]["days"] += day_letter

        order = "MTWRF"
        for r in merged.values():
            r["days"] = "".join(sorted(set(r["days"]), key=lambda x: order.index(x)))

        return list(merged.values())

    def parse(self) -> List[ClassSession]:
        self.header_bottom = self.detect_header_region()
        self.time_col_right = self.detect_time_column()
        h_lines, v_lines = self.detect_grid_lines()
        boxes = self.detect_class_boxes(v_lines)

        for box in boxes:
            text = self.extract_text_from_box(box)
            if len(text) < 5 or any(day.lower() in text.lower() for day in self.DAYS):
                continue

            parsed = self.parse_class_text(text)
            if not parsed["course_code"]:
                continue

            day, start_time, end_time = self.map_box_to_time_day(box, h_lines, v_lines)

            self.classes.append(
                ClassSession(
                    course_code=parsed["course_code"],
                    session_type=parsed["session_type"],
                    building=parsed["building"],
                    room=parsed["room"],
                    day=day,
                    start_time=start_time,
                    end_time=end_time,
                    raw_text=text,
                )
            )

        rows = [asdict(c) for c in self.classes]
        rows = self.merge_rows(rows)
        self.classes = [ClassSession(**r) for r in rows]

        return self.classes

    def to_json(self, filepath: str = None) -> str:
        data = [asdict(c) for c in self.classes]
        json_str = json.dumps(data, indent=2)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
        return json_str


def parse_schedule(image_bytes: bytes, debug: bool = False) -> List[dict]:
    parser = ScheduleParser(image_bytes, debug=debug)
    return parser.parse()
