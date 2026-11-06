# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06  
**Runtime Colab:** CPU/GPU  
**Python / PyTorch / Ultralytics:** Python 3.11 / PyTorch / Ultralytics 8.4.145  
**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`  
**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `day1_lab_outputs/visuals/classification_top5.png`. Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Record hạng 1: `class_id`=468; `class_name`=cab; `rank`=1; `score`=0.510915; `taxonomy_name`=ImageNet-1K.
2. Record này mô tả cả bức ảnh, vì image classification gán một nhãn tổng quát cho toàn khung hình, không định vị vật thể riêng.
3. Danh sách lớp đi kèm taxonomy/checkpoint đã huấn luyện, không phải model tự nghĩ ra khi chạy.
4. Ghi cả `class_id`, `class_name`, `taxonomy_name` giúp xác định lớp chính xác và tránh nhầm giữa taxonomy khác nhau.
5. Guideline cần nêu tiêu chí chọn một chủ đề chính, chẳng hạn vật thể/chủ đề nổi bật nhất, và cách xử lý ảnh có nhiều chủ đề.
6. Không dùng `score` làm ground truth vì đây là độ tin cậy của dự đoán, không phải nhãn chuẩn đã được con người xác nhận.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `day1_lab_outputs/visuals/detection_predictions.png`. Chỉ dùng `sample_id = kitchen`.

1. Ví dụ record: `class_name`=person; `score`=0.912625; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92]; `bbox_width`=113.58; `bbox_height`=279.68.
2. Box có góc trên-trái (385.33, 69.24), góc dưới-phải (498.92, 348.92), rộng 113.58 px và cao 279.68 px.
3. Có 10 prediction ở ngưỡng 0.20 và 6 ở ngưỡng 0.60, tính từ các record kitchen đã lưu (lọc theo score ≥ ngưỡng).
4. Khi hạ ngưỡng, reviewer cần kiểm tra dự đoán sai, trùng lặp hoặc nhiễu và xác nhận nhãn/box.
5. Quy tắc box: ôm sát các điểm cực biên nhìn thấy của đối tượng, không cắt vật thể và không để thừa nhiều nền.
6. Với vật bị che khuất/cắt mép, làm theo quy định thống nhất trong guideline về phần nhìn thấy; escalation nếu không thể nhận diện hoặc xác định ranh giới đáng tin cậy.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `day1_lab_outputs/visuals/segmentation_prediction.png`. Chỉ dùng `sample_id = kitchen`.

1. Ví dụ record từ dữ liệu đã gửi: `instance_id`=kitchen-001; `class_name`=person; `score`=0.899318; `polygon_point_count`=348; ba điểm đầu `polygon_xy`=[[446,70],[445,71],[444,71]].
2. Polygon thể hiện đường biên chi tiết của vật thể; box chỉ là hình chữ nhật bao ngoài.
3. `instance_id` phân biệt từng đối tượng riêng trong kết quả hiện tại; nó không phải class ID hay tracking ID.
4. Quy tắc mask: đặt các đỉnh sát đường viền nhìn thấy, tách từng cá thể và không tô sang nền.
5. Khi hai vật chạm nhau hoặc biên mờ, áp dụng guideline về ranh giới; escalation khi ranh giới không thể xác định nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth (đơn vị nhãn) | Lỗi/chỗ khó | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn lớp cho toàn ảnh | Nhiều đối tượng gây nhiễu chủ đề chính | Chọn nhãn theo guideline | Đối chiếu nhãn với nội dung tổng thể |
| Phát hiện vật thể | Bounding box và class | Che khuất, vật nhỏ hoặc cắt mép | Vẽ box theo quy tắc biên | Kiểm tra vị trí box và nhãn |
| Instance segmentation | Mask/polygon cho từng instance | Biên mờ, vật chạm nhau | Vẽ mask theo đường viền nhìn thấy | Kiểm tra độ khít và phân biệt instance |

## 5. An toàn dữ liệu

1. Tuyệt đối không đưa thông tin cá nhân, ảnh riêng tư hoặc dữ liệu nội bộ nhạy cảm lên Colab hay kho lưu trữ công khai.
2. Nếu gặp dữ liệu ngoài phạm vi, dừng xử lý và báo Lab Coach hoặc giảng viên hướng dẫn.

## 6. Checklist trước khi nộp link repo

- [x] `report/REPORT.md`
- [x] `report/day1_lab_outputs/classification_predictions.json`
- [x] `report/day1_lab_outputs/detection_predictions.json`
- [x] `report/day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `report/day1_lab_outputs/segmentation_predictions.json` (chưa thể tải nguyên văn đầy đủ từ nội dung gửi trong chat)
- [ ] `report/day1_lab_outputs/visuals/classification_top5.png` (cần tải ảnh PNG gốc)
- [ ] `report/day1_lab_outputs/visuals/detection_predictions.png` (cần tải ảnh PNG gốc)
- [ ] `report/day1_lab_outputs/visuals/segmentation_prediction.png` (cần tải ảnh PNG gốc)
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output đã tải lên
- [ ] Đã nộp link repository trên LMS
