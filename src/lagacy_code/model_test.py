"""
YOLOv8 Model Testing & Evaluation Script
สำหรับทดสอบประสิทธิภาพของ model ที่ใช้ตรวจจับ object บนถนน
"""

import os

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import time
from collections import defaultdict
import json

# Fix multiprocessing issue on Windows
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

class YOLOModelTester:
    def __init__(self, model_path, test_data_yaml, output_dir="test_results"):
        """
        Initialize YOLO Model Tester
        
        Args:
            model_path: path ไปยัง model weights (.pt file)
            test_data_yaml: path ไปยัง data.yaml ของ test dataset
            output_dir: folder สำหรับเก็บผลลัพธ์
        """
        self.model = YOLO(model_path)
        self.test_data_yaml = test_data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"✅ โหลด model จาก: {model_path}")
        print(f"✅ Test data: {test_data_yaml}")
        print(f"✅ ผลลัพธ์จะถูกบันทึกใน: {output_dir}")
        
    def run_validation(self, conf=0.25, iou=0.45, split='test'):
        """
        รัน validation และคำนวณ metrics ทั้งหมด
        
        Args:
            conf: confidence threshold (default: 0.25)
            iou: IOU threshold สำหรับ NMS (default: 0.45)
            split: dataset split to use ('test', 'val', or 'valid')
        """
        print("\n" + "="*60)
        print("🚀 เริ่มต้น Model Validation")
        print("="*60)
        
        # รัน validation
        metrics = self.model.val(
            data=self.test_data_yaml,
            split=split,
            conf=conf,
            iou=iou,
            save_json=True,
            plots=True,
            workers=0,  # ปิด multiprocessing เพื่อแก้ปัญหา Windows
            batch=1     # ลด batch size เพื่อประหยัด RAM
        )
        
        # แสดงผล metrics หลัก
        self._print_main_metrics(metrics)
        
        # บันทึกผลลัพธ์
        self._save_metrics(metrics, conf, iou)
        
        return metrics
    
    def _print_main_metrics(self, metrics):
        """แสดงผล metrics หลักๆ"""
        print("\n" + "="*60)
        print("📊 MAIN METRICS")
        print("="*60)
        
        print(f"\n🎯 Overall Performance:")
        print(f"  mAP50     : {metrics.box.map50:.4f}  (ยิ่งใกล้ 1.0 ยิ่งดี)")
        print(f"  mAP50-95  : {metrics.box.map:.4f}  (ยิ่งใกล้ 1.0 ยิ่งดี)")
        print(f"  Precision : {metrics.box.mp:.4f}  (ความแม่นยำ)")
        print(f"  Recall    : {metrics.box.mr:.4f}  (ความครอบคลุม)")
        
        print(f"\n📋 Per-Class Performance:")
        try:
            # แสดง metrics แต่ละ class
            class_names = self.model.names
            maps = metrics.box.maps  # mAP50 แต่ละ class
            
            for i, (class_id, class_name) in enumerate(class_names.items()):
                if i < len(maps):
                    print(f"  {class_name:20s}: mAP50 = {maps[i]:.4f}")
        except Exception as e:
            print(f"  ⚠️ ไม่สามารถแสดง per-class metrics: {e}")
    
    def _save_metrics(self, metrics, conf, iou):
        """บันทึก metrics เป็น JSON"""
        results = {
            "model_path": str(self.model.ckpt_path),
            "test_data": str(self.test_data_yaml),
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "metrics": {
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            },
            "per_class_map50": {}
        }
        
        # เพิ่ม per-class metrics
        try:
            class_names = self.model.names
            maps = metrics.box.maps
            for i, (class_id, class_name) in enumerate(class_names.items()):
                if i < len(maps):
                    results["per_class_map50"][class_name] = float(maps[i])
        except:
            pass
        
        # บันทึกเป็น JSON
        output_file = self.output_dir / "metrics_summary.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 บันทึก metrics summary ที่: {output_file}")
    
    def test_inference_speed(self, test_image_dir, num_images=50):
        """
        ทดสอบความเร็วในการ inference
        
        Args:
            test_image_dir: folder ที่มีรูปภาพ test
            num_images: จำนวนรูปที่ต้องการทดสอบ
        """
        print("\n" + "="*60)
        print("⚡ ทดสอบความเร็ว Inference Speed")
        print("="*60)
        
        image_paths = list(Path(test_image_dir).glob("*.jpg"))[:num_images]
        
        if len(image_paths) == 0:
            print("⚠️ ไม่พบรูปภาพใน folder ที่ระบุ")
            return
        
        times = []
        
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            
            start_time = time.time()
            results = self.model.predict(source=img, conf=0.65, verbose=False)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # แปลงเป็น ms
            times.append(inference_time)
        
        avg_time = np.mean(times)
        avg_fps = 1000 / avg_time
        
        print(f"\n📈 Inference Speed Results (จาก {len(image_paths)} รูป):")
        print(f"  Average Time : {avg_time:.2f} ms")
        print(f"  Average FPS  : {avg_fps:.2f} fps")
        print(f"  Min Time     : {np.min(times):.2f} ms")
        print(f"  Max Time     : {np.max(times):.2f} ms")
        
        if avg_fps >= 25:
            print(f"  ✅ Real-time ready! (≥25 fps)")
        else:
            print(f"  ⚠️ อาจไม่เหมาะสำหรับ real-time (ควร ≥25 fps)")
        
        # บันทึกผลลัพธ์
        speed_results = {
            "num_images_tested": len(image_paths),
            "average_time_ms": float(avg_time),
            "average_fps": float(avg_fps),
            "min_time_ms": float(np.min(times)),
            "max_time_ms": float(np.max(times))
        }
        
        with open(self.output_dir / "inference_speed.json", 'w') as f:
            json.dump(speed_results, f, indent=2)
        
        return avg_fps
    
    def visualize_predictions(self, test_image_dir, num_samples=10, conf=0.65):
        """
        แสดงภาพ predictions พร้อมกับ ground truth
        
        Args:
            test_image_dir: folder รูปภาพ
            num_samples: จำนวนตัวอย่างที่ต้องการแสดง
            conf: confidence threshold
        """
        print("\n" + "="*60)
        print("🖼️  สร้าง Visualization")
        print("="*60)
        
        image_paths = list(Path(test_image_dir).glob("*.jpg"))[:num_samples]
        
        if len(image_paths) == 0:
            print("⚠️ ไม่พบรูปภาพใน folder ที่ระบุ")
            return
        
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        for idx, img_path in enumerate(image_paths):
            img = cv2.imread(str(img_path))
            
            # Predict
            results = self.model.predict(source=img, conf=conf, verbose=False)
            
            # Plot results
            annotated = results[0].plot()
            
            # บันทึกรูป
            output_path = vis_dir / f"prediction_{idx+1}.jpg"
            cv2.imwrite(str(output_path), annotated)
            
            print(f"  ✅ บันทึกรูปที่ {idx+1}/{len(image_paths)}: {output_path.name}")
        
        print(f"\n💾 บันทึกรูป visualizations ทั้งหมดใน: {vis_dir}")
    
    def generate_report(self, metrics, avg_fps=None):
        """
        สร้างรายงานสรุปผล
        """
        print("\n" + "="*60)
        print("📝 สร้างรายงานสรุปผล")
        print("="*60)
        
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("YOLO MODEL EVALUATION REPORT\n")
            f.write("สำหรับการตรวจจับ Object บนถนนเพื่อวิเคราะห์ Collision Risk\n")
            f.write("="*70 + "\n\n")
            
            f.write("1. OVERALL PERFORMANCE\n")
            f.write("-"*70 + "\n")
            f.write(f"mAP50      : {metrics.box.map50:.4f} (IoU threshold = 0.5)\n")
            f.write(f"mAP50-95   : {metrics.box.map:.4f} (IoU threshold = 0.5-0.95)\n")
            f.write(f"Precision  : {metrics.box.mp:.4f}\n")
            f.write(f"Recall     : {metrics.box.mr:.4f}\n")
            
            if avg_fps:
                f.write(f"\n2. INFERENCE SPEED\n")
                f.write("-"*70 + "\n")
                f.write(f"Average FPS: {avg_fps:.2f} fps\n")
                if avg_fps >= 25:
                    f.write("Status     : ✅ Real-time ready\n")
                else:
                    f.write("Status     : ⚠️ อาจไม่เหมาะสำหรับ real-time\n")
            
            f.write(f"\n3. PER-CLASS PERFORMANCE\n")
            f.write("-"*70 + "\n")
            
            try:
                class_names = self.model.names
                maps = metrics.box.maps
                for i, (class_id, class_name) in enumerate(class_names.items()):
                    if i < len(maps):
                        f.write(f"{class_name:20s}: mAP50 = {maps[i]:.4f}\n")
            except:
                f.write("ไม่สามารถแสดง per-class metrics\n")
            
            f.write(f"\n4. RECOMMENDATIONS FOR STAKEHOLDERS\n")
            f.write("-"*70 + "\n")
            
            # ให้คำแนะนำตาม metrics
            if metrics.box.map50 >= 0.8:
                f.write("✅ Model มีความแม่นยำสูง เหมาะสำหรับการใช้งานจริง\n")
            elif metrics.box.map50 >= 0.6:
                f.write("⚠️ Model มีความแม่นยำปานกลาง ควรปรับปรุงเพิ่มเติม\n")
            else:
                f.write("❌ Model ต้องการการปรับปรุงอย่างมาก\n")
            
            if metrics.box.mr < 0.7:
                f.write("⚠️ Recall ต่ำ - model อาจพลาดการตรวจจับ object บางตัว\n")
                f.write("   → มีความเสี่ยงในการพลาดเหตุการณ์อันตราย!\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✅ บันทึกรายงานที่: {report_path}")

def main():
    """
    ตัวอย่างการใช้งาน
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        YOLOv8 Model Testing & Evaluation                     ║
    ║        สำหรับ Collision Risk Analysis                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # ========== ตั้งค่าที่นี่ ==========
    MODEL_PATH = r"models/best.pt"              # path ไปยัง model weights
    TEST_DATA_YAML = r"TestDataset5classes/data.yaml"               # path ไปยัง data.yaml
    TEST_IMAGE_DIR = r"TestDataset5classes/test/images"             # folder รูปภาพ test
    OUTPUT_DIR = "test_results"                # folder สำหรับเก็บผลลัพธ์
    
    CONF_THRESHOLD = 0.65                      # confidence threshold (เหมือนใน video.py)
    IOU_THRESHOLD = 0.45                       # IOU threshold
    # ===================================
    
    # สร้าง tester object
    tester = YOLOModelTester(
        model_path=MODEL_PATH,
        test_data_yaml=TEST_DATA_YAML,
        output_dir=OUTPUT_DIR
    )
    
    # 1. รัน validation
    print("\n🔍 ขั้นตอนที่ 1: Validation")
    metrics = tester.run_validation(
        conf=CONF_THRESHOLD, 
        iou=IOU_THRESHOLD,
        split='test'  # ใช้ 'test' split แทน 'val'
    )
    
    # 2. ทดสอบความเร็ว
    print("\n🔍 ขั้นตอนที่ 2: Speed Test")
    avg_fps = tester.test_inference_speed(
        test_image_dir=TEST_IMAGE_DIR,
        num_images=50
    )
    
    # 3. สร้าง visualizations
    print("\n🔍 ขั้นตอนที่ 3: Visualizations")
    tester.visualize_predictions(
        test_image_dir=TEST_IMAGE_DIR,
        num_samples=10,
        # conf=CONF_THRESHOLD
    )
    
    # 4. สร้างรายงาน
    print("\n🔍 ขั้นตอนที่ 4: Generate Report")
    tester.generate_report(metrics, avg_fps)
    
    print("\n" + "="*60)
    print("✅ เสร็จสิ้น! ตรวจสอบผลลัพธ์ใน folder:", OUTPUT_DIR)
    print("="*60)
    print("\nไฟล์ที่สร้างขึ้น:")
    print("  📄 metrics_summary.json    - สรุป metrics ทั้งหมด")
    print("  📄 inference_speed.json    - ความเร็วในการ inference")
    print("  📄 evaluation_report.txt   - รายงานสรุปผล")
    print("  📁 visualizations/         - รูปภาพ predictions")
    print()

if __name__ == "__main__":
    main()