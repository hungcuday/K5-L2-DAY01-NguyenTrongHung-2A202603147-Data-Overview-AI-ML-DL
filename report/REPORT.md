# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU / GPU (Google Colab)

**Python / PyTorch / Ultralytics:** Python 3.11 / PyTorch 2.x / Ultralytics `8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (giữ nguyên mặc định của notebook)

**Thay đổi so với notebook nguồn:** Không thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `sample_id`=`traffic` (`coco_image_id`=`210273`) ; `class_id`=`468` ; `class_name`=`cab` ; `rank`=`1` ; `score`=`0.510915` ; `taxonomy_name`=`ImageNet-1K` (`model_file`=`yolo11n-cls.pt`).
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?  
   *Trả lời:* Record này nói về **cả bức ảnh** (`image_width = 640`, `image_height = 428`), vì tác vụ `image_classification` chỉ xuất ra thứ hạng (`rank`) và điểm tự tin (`score`) của các lớp cho toàn bộ khung hình mà hoàn toàn không có tọa độ hộp (`bbox_xyxy`) hay đa giác (`polygon_xy`) để định vị bất kỳ vật thể riêng lẻ nào trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?  
   *Trả lời:* Danh sách tên lớp **đi kèm cố định với taxonomy và checkpoint** (`ImageNet-1K` gồm 1000 lớp được lưu sẵn trong `yolo11n-cls.pt`, truy xuất qua `result.names`). Mô hình không thể tự nghĩ ra tên lớp mới ngoài danh sách lớp đã được định nghĩa khi huấn luyện.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?  
   *Trả lời:* Vì các bộ dữ liệu/taxonomy khác nhau có thể dùng tên lớp khác nhau cho cùng một khái niệm hoặc trùng tên lớp nhưng khác mã số `class_id` và phạm vi định nghĩa (ví dụ `ImageNet-1K` chia chi tiết thành `cab` với `class_id = 468`, `minibus` với `class_id = 654`, `police_van` với `class_id = 734`, `streetcar` với `class_id = 829`, trong khi `COCO-80` gộp thành `car` với `class_id = 2` và `bus` với `class_id = 5`). Việc lưu đầy đủ cả `class_id`, `class_name` và `taxonomy_name` giúp truy xuất chính xác, tránh nhầm lẫn khi đối chiếu dữ liệu hoặc tích hợp vào pipeline huấn luyện/đánh giá.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?  
   *Trả lời:* Trên ảnh `traffic` có rất nhiều đối tượng cùng xuất hiện (xe buýt xanh/đỏ, nhiều xe hơi con, xe khách, người đi bộ). Để người gán nhãn chọn **một** lớp duy nhất (single-label) một cách nhất quán, guideline cần quy định rõ **tiêu chí ưu tiên (priority rule)**: ví dụ chọn lớp của vật thể chiếm diện tích lớn nhất/nổi bật nhất ở tiền cảnh, hoặc quy định thứ tự ưu tiên theo nghiệp vụ (ví dụ ưu tiên xe buýt/xe thương mại trước xe con), hoặc quy tắc gắn nhãn bối cảnh chung / đưa lên escalation khi không có vật thể nào chiếm ưu thế.
6. Vì sao không được lấy `score` của model làm ground truth?  
   *Trả lời:* Vì `score` chỉ là độ tự tin thống kê của mô hình dự đoán (model confidence) dùng để xếp hạng hoặc lọc prediction, không phải thước đo sự thật. Bằng chứng rõ ràng trong JSON là ở ảnh `traffic`, lớp `cab` (`score = 0.510915`) được xếp `rank = 1` dù các vật thể lớn và rõ nhất ở tiền cảnh là xe buýt (`bus`) và xe con thường (`car`); thậm chí ở ảnh `kitchen`, model xếp `rank = 1` là `gong` (cái chiêng, `class_id = 577`, `score = 0.420192`) chỉ vì các chảo đồng tròn treo trên tường trông giống cái chiêng. Do đó, ground truth bắt buộc phải do con người tạo và kiểm tra theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `sample_id`=`kitchen` (`coco_image_id`=`397133`) ; `class_id`=`0` ; `class_name`=`person` ; `score`=`0.912625` ; `bbox_xyxy`=`[385.33, 69.24, 498.92, 348.92]` ; `bbox_width`=`113.58` ; `bbox_height`=`279.68`.
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?  
   *Trả lời:* Tính từ gốc tọa độ `(0, 0)` ở góc trên-trái của bức ảnh (`640 x 427` pixel), hộp giới hạn của đối tượng `person` này có **góc trên-trái** nằm tại tọa độ `(x_min = 385.33, y_min = 69.24)` pixel và **góc dưới-phải** nằm tại tọa độ `(x_max = 498.92, y_max = 348.92)` pixel; chiều rộng của hộp là `498.92 - 385.33 = 113.58` pixel và chiều cao của hộp là `348.92 - 69.24 = 279.68` pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?  
   *Trả lời:* Đối chiếu trên ảnh `sample_id = kitchen`:
   - Tại ngưỡng `0.35` (ngưỡng xuất file JSON): có **11 prediction** (`['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl']`, gồm 2 `person`, 5 `bowl`, 2 `oven`, 2 `cup`).
   - Tại ngưỡng `0.60`: còn **6 prediction** (`['person', 'bowl', 'bowl', 'oven', 'oven', 'person']` có `score >= 0.60`; 5 hộp có `score` từ `0.381215` đến `0.499948` gồm 3 `bowl` và 2 `cup` đã bị lọc bỏ).
   - Tại ngưỡng thấp `0.20`: số lượng prediction tăng lên nhiều hơn ngưỡng `0.35` (giữ lại toàn bộ 11 hộp ở mức `0.35` cộng thêm các hộp có độ tự tin thấp trong khoảng `[0.20, 0.35)`).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?  
   *Trả lời:* Khi hạ ngưỡng `conf`, mô hình giữ lại nhiều dự đoán có `score` thấp nên phát sinh thêm nhiều **dương tính giả (false positives)** (nhận diện nhầm lớp), **hộp trùng lặp/đè lên nhau (overlapping/duplicate boxes)** và hộp bao không chuẩn xác. Nếu dùng kết quả này làm gợi ý (pre-annotation), người **reviewer / QC** phải tốn thêm công sức kiểm tra từng hộp để xóa các hộp nhiễu/sai lớp, gộp hoặc loại bỏ hộp trùng lặp và chỉnh lại biên hộp cho khít vật thể thật.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).  
   *Trả lời:* **Quy tắc vẽ Bounding Box:** *"Vẽ một hộp chữ nhật duy nhất cho mỗi đối tượng thuộc danh sách lớp mục tiêu sao cho 4 cạnh của hộp chạm sát các điểm cực biên nhìn thấy được (visible extremes: trái nhất, phải nhất, trên cùng, dưới cùng) của đối tượng; không để thừa khoảng trống nền quá 2 pixel và không cắt lẹm vào phần thân nhìn thấy của vật thể."*
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?  
   *Trả lời:* Trên ảnh `kitchen`, ở sát mép trái có hai trường hợp điển hình:
   - Thứ nhất là cánh tay/bàn tay người thò vào khung hình tại `bbox_xyxy = [0.12, 263.18, 61.55, 310.78]` (`class_name = person`, `score = 0.610931`), bị **cắt mép (truncated)** tại biên trái ảnh (`x_min = 0.12`) và phần lớn cơ thể nằm ngoài ảnh.
   - Thứ hai là các khay/bát trên bàn như `bowl` tại `bbox_xyxy = [0.32, 305.02, 87.69, 371.32]` (`score = 0.381215`) và `bowl` tại `[59.29, 288.92, 133.46, 328.98]` (`score = 0.499948`) vừa bị cắt mép trái vừa bị **che khuất một phần (occluded)** bởi chiếc khuôn/bát phía trước (`[32.65, 342.12, 100.16, 384.93]`).  
   *Cách xử lý:* Với hiện tượng phổ biến, lặp lại trên nhiều ảnh như vật bị che khuất một phần hoặc cắt mép nhưng vẫn nhìn rõ hình dạng/lớp (như chiếc `bowl` hoặc `oven` ở mép trái `[0.11, 187.54, 195.96, 292.81]`), tôi sẽ **đưa vào quy tắc trong guideline** (ví dụ: gán nhãn phần nhìn thấy nếu diện tích hiển thị >= 20% và xác định rõ lớp; bỏ qua nếu bị che khuất > 80%). Ngược lại, với trường hợp cực đoan/mơ hồ như chỉ lộ một bàn tay nhỏ ở mép ảnh (`[0.12, 263.18, 61.55, 310.78]`) mà guideline chưa quy định rõ có tính là một `person` hay không, hoặc các vật dụng nhỏ trên kệ trong bóng tối không phân biệt được là `bowl` hay `cup`, annotator phải **escalation (hỏi Lab Coach)** để thống nhất quy định chung thay vì tự suy đoán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=`kitchen-001` ; `class_name`=`person` (`class_id`=`0`) ; `score`=`0.899318` ; số điểm polygon=`polygon_point_count`=`348` ; 5 điểm đầu của `polygon_xy`=`[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]` (kèm `bbox_xyxy`=`[385.45, 66.44, 498.02, 348.58]`).
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?  
   *Trả lời:* Trong khi bounding box chỉ là một khung chữ nhật thô bao ngoài (chứa cả nhiều pixel nền như khoảng tường, tủ bếp và sàn nhà giữa hai chân của người đầu bếp trong hộp `[385.45, 66.44, 498.02, 348.58]`), thì `polygon_xy` (với `348` đỉnh nối tiếp nhau) cung cấp **đường viền hình học chi tiết bám sát biên dạng thực tế của vật thể** (đầu, vai, cánh tay, tạp dề, hai ống quần), giúp tách chính xác các pixel thuộc về đối tượng khỏi nền và các vật xung quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?  
   *Trả lời:* `instance_id` (như `kitchen-001`, `kitchen-002`, ..., `kitchen-011`) dùng để **định danh duy nhất từng cá thể (instance) riêng biệt** được phát hiện trong phạm vi kết quả của bức ảnh/bài lab đó (nhờ vậy 4 chiếc bát cùng lớp `bowl` vẫn được tách thành 4 cá thể độc lập: `kitchen-002`, `kitchen-003`, `kitchen-006`, `kitchen-010`). Nó **không phải `class_id`** vì `class_id` là mã số của loại vật thể trong taxonomy COCO-80 (ví dụ `person` có `class_id = 0`, `bowl` có `class_id = 45`), và **không phải `tracking_id`** vì `tracking_id` dùng để theo dõi liên tục cùng một đối tượng di chuyển xuyên suốt nhiều khung hình (frames) theo thời gian trong video, còn ở đây chỉ đánh số thứ tự cục bộ trong một ảnh tĩnh.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.  
   *Trả lời:* **Quy tắc vẽ biên Polygon Mask:** *"Đặt các đỉnh đa giác (`polygon_xy`) bám sát đường ranh giới nhìn thấy thực tế của từng cá thể riêng biệt (tăng mật độ điểm ở các đoạn cong/góc gấp như tay, chân, vành bát và giảm điểm ở cạnh thẳng), tuyệt đối không gộp hai vật cùng lớp đứng cạnh nhau vào chung một polygon và không để mask tràn sang pixel nền hoặc vật che khuất phía trước."*
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?  
   *Trả lời:* Quan sát `visuals/segmentation_prediction.png` và JSON của ảnh `kitchen`, có hai chỗ rất rõ:
   - **Hai vật chạm nhau / đè lên nhau:** Nhóm các khay/bát `bowl` (`kitchen-002` có `score = 0.735743`, `kitchen-003` có `score = 0.698698`, `kitchen-006` có `score = 0.528202`) đặt sát và chồng một phần lên nhau trên mặt bàn `dining table` (`kitchen-005`, `score = 0.578416`, `598` điểm); mask của `dining table` bị tràn đè lên vùng bột/đồ vật trên bàn và giao cắt với cánh tay `person` (`kitchen-009`), còn biên giữa `kitchen-002` và `kitchen-006` bị chồng lấn.
   - **Chỗ biên mờ / nhận diện sai do thiếu sáng:** Phần bàn chân của người đầu bếp (`kitchen-001`) chìm vào bóng tối dưới chân tủ bếp, và bó lá khô treo ở góc trên-trái (`kitchen-004`) bị mô hình phân đoạn nhầm thành `potted plant` (`score = 0.632235`, `100` điểm).  
   *Cách xử lý:* Với các vật chạm nhau hoặc đặt chồng lên nhau nhưng nhìn rõ thành/vành vật thể (như các khay `bowl` đặt trên `dining table`), tôi áp dụng **guideline** để vẽ tách rời từng polygon theo phần diện tích nhìn thấy của mỗi cá thể và khoét trừ phần bị vật phía trước che khuất. Với vùng biên chìm hoàn toàn vào bóng tối không thể phân biệt ranh giới pixel bằng mắt thường hoặc vật thể lạ ngoài dự kiến (như bó lá treo tường có được tính là `potted plant` hay không), tôi sẽ **escalation (hỏi Lab Coach)** vì tự suy đoán sẽ tạo ra ground truth không nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp (`class_id`, `class_name`) thuộc taxonomy gán cho **toàn bộ bức ảnh** (không có tọa độ vị trí). | Trên `traffic` (`classification_top5.png`), ảnh có cả `bus`, `car`, `person` nhưng model chỉ gán `rank=1` là `cab` (`class_id=468`, `score=0.510915`); trên `kitchen`, model đoán nhầm cả căn bếp thành cái chiêng `gong` (`class_id=577`, `score=0.420192`) do các chảo đồng treo tường. | Quan sát toàn bộ ảnh, áp dụng quy tắc chọn chủ đề chính trong guideline để gán đúng nhãn cấp ảnh (không chép `score`/`rank=1` của model); báo escalation nếu ảnh đa chủ đề chưa có rule. | Kiểm tra nhãn cấp ảnh có phản ánh đúng chủ đề chính theo guideline và đúng taxonomy không; bắt lỗi các ca dễ nhầm thị giác như `kitchen` hoặc đa đối tượng như `traffic`. |
| Phát hiện vật thể | Mỗi vật thể riêng biệt là **một hộp chữ nhật** `bbox_xyxy = [x_min, y_min, x_max, y_max]` (pixel) kèm `class_id`, `class_name`. | Trên `kitchen` (`detection_predictions.png`), bàn tay cắt mép trái `[0.12, 263.18, 61.55, 310.78]` vẫn được gán `person` (`score=0.610931`); các hộp `bowl` (`[32.65, 342.12, 100.16, 384.93]` và `[0.32, 305.02, 87.69, 371.32]`) đè lên nhau; nhiều vật thật (chảo treo tường, bàn ăn, lọ trên bàn) bị bỏ sót ở ngưỡng `0.35`. | Vẽ bounding box ôm sát phần nhìn thấy của từng vật thể thuộc lớp mục tiêu, bổ sung vật bị model bỏ sót (false negatives), xóa box sai/trùng và tuân thủ rule cắt mép/che khuất. | Kiểm tra độ khít 4 cạnh box (không thừa nền, không cắt lẹm vật), đúng `class_name`, không sót object, không dư box trùng lặp và xử lý đồng nhất các ca che khuất/cắt mép. |
| Instance segmentation | Mỗi cá thể riêng biệt là **một đa giác khép kín** `polygon_xy = [[x, y], ...]` bám sát đường viền pixel kèm `class_id`, `class_name` và định danh instance riêng. | Trên `kitchen` (`segmentation_prediction.png`), bó lá treo góc trái (`kitchen-004`) bị đoán nhầm là `potted plant` (`score=0.632235`); mask `dining table` (`kitchen-005`, `598` điểm) tràn đè lên đồ vật trên bàn; các khay `bowl` (`kitchen-002`, `kitchen-003`, `kitchen-006`) chạm nhau có viền chồng lấn; `kitchen-010` (`bowl`) chỉ có `12` điểm khá thô. | Đặt các đỉnh polygon bám sát biên thật của từng cá thể, tách riêng từng instance cùng lớp (`bowl`, `spoon`), xóa nhãn sai (`potted plant`) và không để mask tràn sang nền/vật kề cạnh. | Kiểm tra đường viền polygon có bám khít biên vật thể không, các vật cùng lớp chạm nhau đã tách đúng từng instance riêng chưa, và có bị lẹm hoặc tràn mask ở vùng tối/giao nhau không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
   *Trả lời:* Chỉ sử dụng ba ảnh COCO công khai có kiểm tra checksum và ghi nhận bản quyền (`IMAGE_ATTRIBUTION.md`), tuyệt đối không tải ảnh cá nhân, dữ liệu nội bộ/khách hàng và không ghi họ tên, MSSV, email, số điện thoại hay token vào `REPORT.md` hoặc các tệp đầu ra công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
   *Trả lời:* Dừng thao tác ngay lập tức, không tải lên hoặc chia sẻ tiếp, và báo cáo trực tiếp cho **Lab Coach** (hoặc giảng viên hướng dẫn) để được xử lý theo quy trình.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
