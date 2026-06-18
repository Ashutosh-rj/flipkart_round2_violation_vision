import sys
import os
import pytest

# Add backend directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_pipeline import ViolationDetector

@pytest.fixture
def detector():
    return ViolationDetector()

def test_is_rider_on_motorcycle(detector):
    # Format: [x1, y1, x2, y2]
    person_box = [100, 100, 150, 200]
    moto_box = [80, 150, 180, 250]
    # Intersects significantly, should be true
    assert detector.is_rider_on_motorcycle(person_box, moto_box) == True

    # No intersection
    person_box2 = [10, 10, 50, 50]
    assert detector.is_rider_on_motorcycle(person_box2, moto_box) == False

def test_classify_severity(detector):
    # Triple Riding
    assert detector.classify_severity("Triple Riding", rider_count=4, confidence=0.9) == "CRITICAL"
    assert detector.classify_severity("Triple Riding", rider_count=3, confidence=0.8) == "MAJOR"

    # Helmet Non-compliance
    assert detector.classify_severity("Helmet Non-compliance", rider_count=2, confidence=0.9) == "MAJOR"

    # Wrong-side Driving
    assert detector.classify_severity("Wrong-side Driving", rider_count=0, confidence=0.9) == "CRITICAL"

def test_apply_nms(detector):
    boxes = [
        [100, 100, 200, 200],  # Area = 10000
        [105, 105, 195, 195],  # Highly overlapping with first box
        [300, 300, 400, 400],  # Distinct box
    ]
    scores = [0.9, 0.8, 0.95]
    
    # NMS should filter out the second box (index 1)
    keep = detector.apply_nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 2
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep

def test_compute_composite_confidence(detector):
    moto_conf = 0.8
    person_confs = [0.9, 0.7]
    # composite = 0.4 * 0.8 + 0.6 * (1.6 / 2) = 0.32 + 0.48 = 0.8
    assert detector.compute_composite_confidence(moto_conf, person_confs) == 0.8

    # Edge case: no persons
    assert detector.compute_composite_confidence(moto_conf, []) == 0.8
