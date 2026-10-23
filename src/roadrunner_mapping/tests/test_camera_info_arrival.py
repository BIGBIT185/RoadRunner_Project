"""Run actual node callbacks without constructing GPU/ROS workers."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace as NS
from roadrunner_mapping.online.online_backend import BoundedCameraInfoCache


def make_node():
    path = Path(__file__).resolve().parents[1] / 'scripts/mapping_node.py'
    tree = ast.parse(path.read_text())
    methods = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            methods += [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name in {
                '_camera_info_callback', '_synchronized_callback', '_submit_semantic_sensor'}]
    cls = ast.ClassDef(name='Callbacks', bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
    ns = {'stamp_to_nsec':lambda s:s, 'time':NS(monotonic=lambda:10.), 'rospy':NS(loginfo_once=lambda *a:None)}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
    n = ns['Callbacks']()
    n.received=0; n.stream_epoch=0; n.last_input_stamp_ns=None
    n.stream_lock=threading.Lock(); n.camera_info_lock=threading.Lock()
    n.camera_info_cache=BoundedCameraInfoCache()
    n.pending_camera_sensor=None; n.camera_info_wait_sec=.5
    n.camera_info_pending_dropped=0; n.camera_info_late_matched=0
    n.semantic_matcher=None; n.semantic_enabled=True; n.semantic_backend='deeplab'
    n.geometry=[]; n.semantic=[]
    n._enqueue_geometry=n.geometry.append; n._enqueue_semantic=n.semantic.append
    return n


def message(stamp):return NS(header=NS(stamp=stamp))


def test_late_camera_info_recovers_semantics_without_requeuing_geometry():
    n=make_node(); image=message(1000000000); cloud=message(1000000000); info=message(1000000000)
    n._synchronized_callback(image,cloud)
    assert len(n.geometry)==1 and not n.semantic
    n._camera_info_callback(info)
    assert len(n.geometry)==1 and len(n.semantic)==1
    assert n.semantic[0][3] is info


def test_early_camera_info_and_duplicate_arrival_do_not_duplicate_work():
    n=make_node(); m=message(1000000000)
    n._camera_info_callback(m); n._synchronized_callback(m,m); n._camera_info_callback(m)
    assert len(n.geometry)==len(n.semantic)==1


def test_only_latest_waiting_frame_is_retained():
    n=make_node(); old=message(1000000000); new=message(2000000000)
    n._synchronized_callback(old,old); n._synchronized_callback(new,new)
    n._camera_info_callback(old); assert not n.semantic
    n._camera_info_callback(new)
    assert len(n.geometry)==2 and len(n.semantic)==1
    assert n.semantic[0][1] is new


def test_expired_waiting_frame_is_not_revived():
    n=make_node(); m=message(1000000000); n._synchronized_callback(m,m)
    n.camera_info_wait_sec=0
    n._camera_info_callback(m)
    assert not n.semantic


def test_previous_epoch_late_sensor_is_rejected():
    n=make_node(); m=message(1000000000)
    n.stream_epoch=1
    n._submit_semantic_sensor(0,m,m,m)
    assert not n.semantic


def test_geometry_only_does_not_retain_waiting_images():
    n=make_node(); n.semantic_enabled=False; m=message(1000000000)
    n._synchronized_callback(m,m)
    assert len(n.geometry)==1 and n.pending_camera_sensor is None
