from argparse import ArgumentParser
from pathlib import Path
import json
import re
from itertools import chain

import math
import warnings

from typing import Union, Tuple, List, Dict, Any


def strip_upload_prefix_from_filename(filename_with_prefix: str) -> str:
    """
    ignores the preceding hash that Label Studio automatically adds to uploaded files
    :param filename_with_prefix: name of the file in LabelStudio
    :return: original name of the file (without the unique hash attached to before the name by LabelStudio)
    """
    # strip the preceding hash code
    re_prefix = re.compile("[a-z0-9]+-")
    m = re_prefix.match(filename_with_prefix)
    if m:
        filename = filename_with_prefix[m.end() :]
    else:
        filename = filename_with_prefix
    return filename


def to_decimal_precision(
        num: Union[float, int],
        precision: int = 5
) -> Union[float, int]:
    out = round(num, precision)
    if precision == 0:
        out = int(num)
    return out


def rotate_point(point, angle, center):
    """Rotate a point around a center by a given angle."""
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    x_shifted = point[0] - center[0]
    y_shifted = point[1] - center[1]
    x_new = x_shifted * cos_theta - y_shifted * sin_theta + center[0]
    y_new = x_shifted * sin_theta + y_shifted * cos_theta + center[1]
    return [x_new, y_new]


def get_non_rotated_bbox(center, width, height, angle):
    """Calculate the non-rotated bounding box from a rotated bounding box."""
    half_width = width / 2
    half_height = height / 2
    # Calculate the corner points of the rotated bounding box
    rotated_bbox = [
        [center[0] - half_width, center[1] - half_height],  # Top-left
        [center[0] + half_width, center[1] - half_height],  # Top-right
        [center[0] + half_width, center[1] + half_height],  # Bottom-right
        [center[0] - half_width, center[1] + half_height]   # Bottom-left
    ]
    # Rotate each corner point back to the original orientation
    rotated_points = [rotate_point(point, -angle, center) for point in rotated_bbox]
    # Find min and max x, y coordinates to form the non-rotated bounding box
    min_x = min(rotated_points, key=lambda x: x[0])[0]
    max_x = max(rotated_points, key=lambda x: x[0])[0]
    min_y = min(rotated_points, key=lambda x: x[1])[1]
    max_y = max(rotated_points, key=lambda x: x[1])[1]
    # non_rotated_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
    # Calculate center of the non-rotated bounding box
    non_rotated_center = [(min_x + max_x) / 2, (min_y + max_y) / 2]
    # Calculate width and height of the non-rotated bounding box
    non_rotated_width = max_x - min_x
    non_rotated_height = max_y - min_y
    return non_rotated_center, non_rotated_width, non_rotated_height


class LabelStudioExporter:
    _decimal_precision = 5

    # {
    #   "id": int,
    #   "annotations": List[{
    #       "id": int,
    #       "completed_by": int,
    #       "result": List[{
    #           "id": str,
    #           "type": str,
    #           "value": {
    #               "closed": bool,
    #               "points": List[List[float]],
    #               "polygonlabels": List[str]
    #               },
    #           "origin": str,
    #           "to_name": str,
    #           "from_name": str,
    #           "image_rotation": int,
    #           "original_width": int,
    #           "original_height": int
    #       }],
    #       "was_cancelled": bool,
    #       "ground_truth": bool,
    #       "created_at": string,
    #       "updated_at": string,
    #       "draft_created_at": ?,
    #       "lead_time": float,
    #       "prediction": {},
    #       "result_count": int,
    #       "unique_id": string,
    #       "last_action": ?,
    #       "task": int,
    #       "project": int,
    #       "updated_by": int,
    #       "parent_prediction": ?,
    #       "parent_annotation": ?,
    #       "last_created_by": ?
    #   }],
    #   "file_upload": string,
    #   "drafts: List[?],
    #   "predictions": List[?],
    #   "data": {
    #       "image": string,
    #       "file_upload": string
    #       },
    #   "meata": {?},
    #   "created_at": string,
    #   "updated_at": string,
    #   "inner_id": int,
    #   "total_annotations": int,
    #   "cancelled_annotations": int,
    #   "total_predictions": int,
    #   "comment_count": int,
    #   "unresolved_comment_count": int,
    #   "last_comment_updated_at": string,
    #   "project": int,
    #   "updated_by": int,
    #   "comment_authors": List[?]
    # }

    def __init__(
        self,
        annotations: Union[dict, str, Path],
        labels_to_exclude: Union[List[str], Dict[str, str]] = None,
        label_mapping: Dict[str, Dict[str, int]] = None
    ) -> None:
        if isinstance(annotations, dict):
            self.data = annotations
        elif isinstance(annotations, (str, Path)):
            # ensure pathlib object
            path_to_annotations = Path(annotations)
            if not path_to_annotations.exists():
                raise Exception(
                    f"Annotation file {path_to_annotations.as_posix()} does not exist!"
                )
            # load file
            with open(path_to_annotations, "r", encoding="utf-8") as fid:
                self.data = json.load(fid)
            # set path (for __repr__ only)
            self._path_to_annotations = path_to_annotations
        else:
            raise Exception(
                f"LabelStudioExporter expects a dictionary or the path to a JSON file as input, "
                f"but input was of type {type(annotations)}."
            )
        self._labels_to_exclude = labels_to_exclude if isinstance(labels_to_exclude, list) else []

        # process data, create mappings
        mappings = dict()
        mappings["task2annotation_type"] = self.__task2annotation_type_map()
        mappings["annotation_type2task"] = self.__annotation_type2task_map(
            mappings["task2annotation_type"]
        )
        labelcategories = self.__label2int(labels_to_exclude)
        if label_mapping is not None:
            for an_typ, map_ in label_mapping.items():
                for name, code in map_.items():
                    labelcategories[an_typ][name] = code
        mappings["labelcategories"] = labelcategories

        self._map = mappings

    # ----- magic functions
    def __repr__(self) -> str:
        if self._path_to_annotations:
            txt = self._path_to_annotations.as_posix()
        else:
            txt = str(self.data)
        return f"LabelStudioExporter({txt})"

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def __len__(self, annotation_type: Union[str, List[str]] = None) -> int:
        return len(self._map["annotation_type2task"][annotation_type])

    # ----- helper functions
    # def _istype(self, annotation_type: str, expected_type: Union[str, List[str], None]) -> bool:
    #     def match(pattern: str):
    #         return re.match(pattern, annotation_type, re.IGNORECASE)
    #
    #     if expected_type is None:
    #         return True
    #     elif isinstance(expected_type, str):
    #         return True if match(expected_type) else False
    #     elif isinstance(expected_type, (list, tuple)):
    #         return any([match(el) for el in expected_type])
    #     else:
    #         raise Exception("Unexpected input type.")

    # def _set_decimal_precision(self, decimal_precision: int) -> bool:
    #     # store precision value (if it is valid)
    #     self._decimal_precision = decimal_precision if decimal_precision > 0 else 0
    #     return True

    def _to_decimal_precision(
        self, num: Union[float, int], precision: int = None
    ) -> Union[float, int]:
        if precision is None:
            precision = self._decimal_precision
        return to_decimal_precision(num, self._decimal_precision)

    def _itertasks(
        self, annotation_type: Union[str, List[str]] = None
    ) -> Union[Dict[str, Any], None]:
        # label-studio organizes annotations per tasks / files => loop over all files
        if annotation_type in self._map["annotation_type2task"]:
            for idx in self._map["annotation_type2task"][annotation_type]:
                yield self.data[idx]
        else:
            warnings.warn(f"No annotations of type '{annotation_type}' found.")
            return None

    # ----- mappings
    def __task2annotation_type_map(self) -> List[List[str]]:
        annotation_types_per_task = []
        for tsk in self.data:
            an_types = []
            for an in tsk["annotations"]:
                # list of unique types of this annotation
                uq = list(set([el["type"] for el in an["result"]]))
                an_types.append(uq)
            annotation_types_per_task.append(an_types)
        return annotation_types_per_task

    def __annotation_type2task_map(
        self, annotation_types_per_task: List[List[str]]
    ) -> Dict[str, List[int]]:
        # unique annotation types
        annotation_types = set(
            chain.from_iterable(
                [
                    list(chain.from_iterable(an_tp))
                    for an_tp in annotation_types_per_task
                ]
            )
        )
        # create map from annotation type to tasks
        annotation_type_map = dict()
        for tp in annotation_types:
            # find all tasks that contain this annotation type
            tasks_per_annotation_type = []
            for i, tsk_tp in enumerate(annotation_types_per_task):
                if tp in chain.from_iterable(tsk_tp):
                    tasks_per_annotation_type.append(i)
            annotation_type_map[tp] = tasks_per_annotation_type

        annotation_type_map[None] = list(range(len(self.data)))
        return annotation_type_map

    def __label2int(
        self, labels_to_exclude: Union[List[str], Dict[str, List[str]]] = None
    ):
        label2int: Dict[str, Dict[str, int]] = dict()

        for tsk in self.data:
            for an in tsk["annotations"]:
                for res in an["result"]:
                    # label type
                    an_type = res["type"]
                    # actual label (points + meta data)
                    value = res["value"]
                    # name of the label
                    if an_type not in value:
                        continue
                    label_name = value[an_type]

                    # add to mapping
                    if an_type not in label2int:
                        label2int[an_type] = dict()

                    for nm in label_name:
                        if nm not in label2int[an_type]:
                            add_nm = True
                            if labels_to_exclude is not None:
                                if isinstance(labels_to_exclude, list):
                                    if nm in labels_to_exclude:
                                        add_nm = False
                                elif isinstance(labels_to_exclude, dict):
                                    if (
                                        an_type in labels_to_exclude
                                        and nm in labels_to_exclude[an_type]
                                    ):
                                        add_nm = False

                            if add_nm:
                                id = len(label2int[an_type])
                                label2int[an_type][nm] = id

        return label2int

    # ----- file name of original image
    @staticmethod
    def _get_original_filename(file: dict) -> str:
        if "data" in file:
            data = file["data"]
            if "image" in data:  # TODO: extend to other data types
                filename = Path(file["data"]["image"]).name
                return strip_upload_prefix_from_filename(filename)
            else:
                raise Exception
        else:
            raise Exception

    def get_filename(
        self, task: dict = None, task_id: int = None, idx: int = None
    ) -> str:
        if isinstance(task, dict):
            pass
        elif task_id >= 0:
            task = self.get_task(task_id)
        elif idx >= 0:
            task = self[idx]
        else:
            raise Exception(
                "No appropriate input provided. "
                "Expected input 'task' to be a dictionary; "
                "'task_id' and 'idx' to be a natural number. "
                "Only one of the inputs should be provided."
            )

        return self._get_original_filename(task)

    def get_categories(
        self, key: str = None
    ) -> Union[Dict[str, Dict[str, List[str]]], Dict[str, List[str]]]:
        return (
            self._map["labelcategories"][key]
            if key and key in self._map
            else self._map["labelcategories"]
        )

    # ----- get task / item / label
    def get_item(self, idx: int, annotation_type: Union[str, List[str]] = None):
        n = self.__len__(annotation_type)
        if idx < n:
            return self.data[self.__map["annotation_type2task"][annotation_type][idx]]
        else:
            msg = f"Index out of bounds! There are only {n} tasks"
            if annotation_type:
                msg += f" of type {annotation_type}"
            raise Exception(msg + ".")

    def get_task(self, task_id: int) -> dict:
        if self.__task_to_idx_mapping is None:
            self.__task_to_idx_mapping = {
                tsk: i for i, tsk in enumerate(self._itertasks())
            }

        if task_id not in self.__task_to_idx_mapping:
            raise Exception(f"Task id {task_id} not in annotations.")
        return self.data[self.__task_to_idx_mapping[task_id]]

    def _get_labels(
        self, annotation_type: Union[str, List[str]] = None
    ) -> Tuple[List[Dict[str, Union[str, None, List[float]]]], str]:
        for tsk in self._itertasks(annotation_type):
            filename = self.get_filename(tsk)

            labels = []
            # loop over all annotators
            for an in tsk["annotations"]:
                if any([any([x in self._labels_to_exclude for x in el["value"][el["type"]]]) for el in an["result"]]):
                    continue
                else:
                    for names, an_type, res in self.__get_results(an, annotation_type):
                        if len(names) > 1:
                            warnings.warn(f"Multiple label names: {names}! Only using the first one.")
                        for nm in names:
                            if nm not in self._labels_to_exclude:
                                labels.append({"name": nm, "type": an_type, "data": res})
                    break
                    # TODO: select or merge multiple annotations per task
            yield labels, filename

    def get_labels(
        self,
            annotation_type: Union[str, List[str]] = None,
            label_style: str = "yolo"
    ) -> Dict[str, List[str]]:
        if isinstance(annotation_type, str):
            if annotation_type.lower() == "choices":
                # loop over annotations
                labels: Dict[str, List[str]] = dict()
                for i, (lbls, fl_nm) in enumerate(self._get_labels(annotation_type)):
                    for lbl in lbls:
                        name = lbl["name"]
                        if name not in labels:
                            labels[name] = []
                        if fl_nm not in labels[name]:
                            labels[name].append(fl_nm)

            elif annotation_type.lower() in ["polygonlabels", "rectanglelabels"]:
                # loop over annotations
                labels: Dict[str, List[str]] = dict()
                for i, (lbls, fl_nm) in enumerate(self._get_labels(annotation_type)):
                    lines: List[str] = []
                    for lbl in lbls:
                        mapping = self._map["labelcategories"][lbl["type"]]
                        if lbl["name"] in mapping:
                            # lookup class id
                            class_id = mapping[lbl["name"]]
                            # flatten list of coordinates
                            if annotation_type.lower() == "polygonlabels":
                                points = chain.from_iterable(lbl["data"])
                            else:
                                points = lbl["data"]

                            if label_style.lower() == "yolo":
                                if annotation_type.lower() == "rectanglelabels":
                                    points = (
                                            points[0] + points[2] / 2,  # from lower left corner to center
                                            points[1] + points[3] / 2,  # from lower left corner to center
                                            points[2], points[3]
                                    )
                                # scale points and make string
                                points_str = [f"{el / 100:0.5f}" for el in points]
                                lines.append(" ".join([str(class_id)] + points_str))
                            else:
                                ValueError(f"Export style {label_style} not yet implemented.")
                        else:
                            # reset
                            lines = []
                            break
                    if lines:
                        labels[fl_nm] = lines
            else:
                raise ValueError(
                    f"Annotation type {annotation_type} not yet implemented for export."
                )
        else:
            raise ValueError(
                f"A single annotation type must be specified to convert labels."
            )

        return labels

    def __is_an_type(
        self, an_type: str, annotation_type: Union[str, List[str]] = None
    ) -> bool:
        """helper function to wrap the comparison of different input types"""
        if annotation_type is None:
            return True
        elif isinstance(annotation_type, str):
            return an_type == annotation_type
        elif isinstance(annotation_type, list):
            return an_type in annotation_type
        else:
            raise Exception

    def __get_results(
        self, annotation: dict, annotation_type: Union[str, List[str]] = None
    ):
        for res in annotation["result"]:
            an_type = res["type"]
            if self.__is_an_type(an_type, annotation_type):
                # extract value
                value = res["value"]
                if an_type not in value:
                    # probably a deleted type
                    continue
                label_name = value[an_type]
                points = None

                if an_type == "polygonlabels":
                    if value["closed"]:
                        points = value["points"]
                        yield label_name, an_type, points
                    else:
                        points = None
                elif an_type == "choices":
                    yield label_name, an_type, None
                elif an_type == "rectanglelabels":
                    if value["rotation"] != 0:
                        msg = f"Rectangles should not be rotated but rotation was {round(value['rotation'])}°. (Task {annotation['task']}"
                        # raise Exception(msg)
                        warnings.warn(msg)

                        (x, y), w, h = get_non_rotated_bbox(
                            (value["x"], value["y"]),
                            value["width"],
                            value["height"],
                            value['rotation']
                        )

                        value["x"] = x
                        value["y"] = y
                        value["width"] = w
                        value["height"] = h
                    # x, y represent lower left corner of the bounding box.
                    yield label_name, an_type, (value["x"], value["y"], value["width"], value["height"])
                else:
                    print(f"Annotation type not yet implemented: {an_type}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, help="JSON file with Label-Studio exports.")
    parser.add_argument("--destination", type=str, default="",
                        help="Directory to export the labels to")

    parser.add_argument("--labels-to-exclude", type=str, nargs="+", default=None, help="Names of labels that should be excluded.")
    parser.add_argument("--label-mapping", type=str, default=None, help="Mapping of label names as JSON string")

    parser.add_argument("--type", type=str, default="rectanglelabels", help="Annotation type. Can be 'choices', 'rectanglelabels', or 'polygonlabels'.")
    parser.add_argument("--label-style", type=str, default="yolo", help="Format style of the exported labels.")

    opt = parser.parse_args()

    print(opt)

    # process input
    path = Path(opt.source)
    export_dir = Path(opt.destination)

    labels_to_exclude = opt.labels_to_exclude

    # label / class mapping
    label_mapping = opt.label_mapping
    if isinstance(label_mapping, str):
        try:
            label_mapping = json.loads(label_mapping)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for argument --label-mapping: {e}")
    # annotation type
    annotation_type = opt.type
    # format style
    label_style = opt.label_style


    exporter = LabelStudioExporter(path, label_mapping=label_mapping, labels_to_exclude=labels_to_exclude)
    labels = exporter.get_labels(annotation_type=annotation_type, label_style=label_style)

    print(exporter.get_categories(annotation_type))

    if not export_dir.exists():
        export_dir.mkdir(parents=True)

    for fl, txt in labels.items():
        filename = re.sub("[\!\?]", "", fl).strip("")
        with open((export_dir / filename).with_suffix(".txt"), "w") as fid:
            for ln in txt:
                fid.write(ln + "\n")
    print(f"Done writing {len(labels)} files to {export_dir.as_posix()}.")
