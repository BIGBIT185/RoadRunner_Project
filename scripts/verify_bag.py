#!/usr/bin/env python3
"""Observe a live bag replay using wall time; write evidence and exit nonzero on failure."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import rospy
from diagnostic_msgs.msg import DiagnosticArray
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Bool
from roadrunner_mapping.msg import EomtSemanticPrediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.)
    parser.add_argument("--output", default="reports/goose_bag_validation.json")
    args = parser.parse_args()
    rospy.init_node("roadrunner_bag_verification", anonymous=True)
    lock = threading.Lock()
    result = dict(inputs={}, maps={}, semantic_predictions=0, semantic_ready=False,
                  mapper_diagnostics={}, semantic_diagnostics={}, errors=[])
    latest = {}

    def count(name):
        def callback(message):
            with lock:
                result["inputs"][name] = result["inputs"].get(name, 0) + 1
        return callback

    def grid(name):
        def callback(message):
            with lock:
                arrays = {k: np.array(v.data, np.float32)
                          for k, v in zip(message.layers, message.data)}
                summary = result["maps"].setdefault(name, dict(count=0))
                summary["count"] += 1
                summary.update(frame=message.info.header.frame_id,
                               stamp=message.info.header.stamp.to_sec(),
                               resolution=message.info.resolution,
                               length_x=message.info.length_x,
                               length_y=message.info.length_y,
                               layers={})
                for layer, values in arrays.items():
                    finite = values[np.isfinite(values)]
                    summary["layers"][layer] = dict(
                        cells=int(values.size), finite=int(finite.size),
                        min=float(finite.min()) if finite.size else None,
                        max=float(finite.max()) if finite.size else None)
                    if "probability" in layer and finite.size and (
                            float(finite.min()) < -1e-6 or float(finite.max()) > 1.000001):
                        result["errors"].append(f"{name}/{layer}: probability outside [0,1]")
                if "elevation" in arrays and "geometry_probability" in arrays:
                    invalid = ~np.isfinite(arrays["geometry_probability"])
                    if np.isfinite(arrays["elevation"][invalid]).any():
                        result["errors"].append(name + ": unknown geometry has finite elevation")
                latest[name] = arrays
        return callback

    def prediction(message):
        with lock:
            result["semantic_predictions"] += 1
            result["prediction_shape"] = [message.labels.height, message.labels.width]
            result["last_inference_sec"] = message.inference_sec
            result["prediction_stamp"] = message.header.stamp.to_sec()

    def ready(message):
        with lock:
            result["semantic_ready"] = bool(message.data)

    def diagnostics(name):
        def callback(message):
            with lock:
                result[name] = {pair.key: pair.value
                                for status in message.status for pair in status.values}
                result[name + "_status"] = [dict(level=s.level, message=s.message)
                                            for s in message.status]
        return callback

    subscribers = []
    for key, kind in [("image_topic", Image), ("cloud_topic", PointCloud2),
                      ("camera_info_topic", CameraInfo)]:
        topic = rospy.get_param("/roadrunner/" + key)
        subscribers.append(rospy.Subscriber(topic, kind, count(key), queue_size=1))
    for key, topic in [("ordinary", "/roadrunner/traversability_probability"),
                       ("cvar", "/roadrunner/cvar_traversability_probability")]:
        subscribers.append(rospy.Subscriber(topic, GridMap, grid(key), queue_size=2))
    subscribers += [
        rospy.Subscriber("/roadrunner_semantic/prediction", EomtSemanticPrediction, prediction, queue_size=1),
        rospy.Subscriber("/roadrunner_semantic/ready", Bool, ready, queue_size=1),
        rospy.Subscriber("/roadrunner/diagnostics", DiagnosticArray, diagnostics("mapper_diagnostics"), queue_size=1),
        rospy.Subscriber("/roadrunner_semantic/diagnostics", DiagnosticArray, diagnostics("semantic_diagnostics"), queue_size=1),
    ]
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline and not rospy.is_shutdown():
        time.sleep(.1)
    with lock:
        for key in ["image_topic", "cloud_topic", "camera_info_topic"]:
            if not result["inputs"].get(key): result["errors"].append("No input: " + key)
        for key in ["ordinary", "cvar"]:
            summary = result["maps"].get(key, {})
            if summary.get("count", 0) < 3: result["errors"].append("Insufficient maps: " + key)
            for layer in ["elevation", "final_probability", "semantic_probability"]:
                if not summary.get("layers", {}).get(layer, {}).get("finite", 0):
                    result["errors"].append("No finite layer: " + key + "/" + layer)
        if not result["semantic_ready"] or not result["semantic_predictions"]:
            result["errors"].append("Semantic model/predictions unavailable")
        if int(result["mapper_diagnostics"].get("semantic_processed", 0)) < 1:
            result["errors"].append("No semantic fusion processed")
        result["passed"] = not result["errors"]
        result["duration_wall_sec"] = args.duration
        result["ros_master_uri"] = rospy.get_master().getUri()[2]
        result["code_origins"] = {name: importlib.import_module(name).__file__
                                  for name in ["roadrunner_mapping.geometry.gaussian_elevation",
                                               "roadrunner_mapping.online.online_fusion"]}
        result["old_goose_package_imported"] = any(k.startswith("goose_roadrunner_dataset") for k in sys.modules)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if latest:
            np.savez_compressed(output.with_suffix(".npz"), **{
                name + "_" + layer: data for name, layers in latest.items()
                for layer, data in layers.items()})
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
