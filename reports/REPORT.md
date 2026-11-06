# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:11/06/2026**

**Runtime Colab:** GPU

**Python:3.13.16 / PyTorch:2.11.0+CU130 / Ultralytics:8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)
|Từ|Nghĩa ngắn|
|-|-|
|Prediction|Kết quả model vừa đoán ra; chưa phải nhãn đúng|
|Ground truth|Nhãn do người gán theo quy tắc|
|Taxonomy|Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80)|
|Checkpoint|File model `.pt` đã huấn luyện sẵn|
|Guideline|Quy tắc gán nhãn do người viết|
|Annotator|Người gán nhãn|
|Reviewer / QC|Người kiểm tra nhãn|
|Escalation|Ca khó → hỏi Lab Coach, không tự bịa|
|Model score|Số “tin” của model; không phải điểm chất lượng nhãn người|

## 1\. Phân loại ảnh
1. `class\_id`=468 ; `class\_name`=cab ; `rank`=1 ; `score`=0.510915; `taxonomy\_name`="ImageNet-1K"
2. record này nói về cả bức ảnh vì trường task ghi là image_classification 
3. danh sách tên lớp là đi kèm với checkpoint của mô hình 
classification_model = YOLO(CLASSIFICATION_MODEL_FILE)
tên của từng lớp được lấy ra từ "class_name": names[int(class_id)]
dữ liệu phân loại cũng ghi rõ trường "taxonomy_name": "ImageNet-1K".

4. giúp đảm bảo tính chính xác trong quản lý dữ liệu
taxonomy_name: Xác định rõ bộ từ điển/tiêu chuẩn nhãn đang được sử dụng.
class_id: Là mã định danh bằng số duy nhất.
class_name: Giúp dễ dàng đọc và hiểu được kết quả một cách trực quan (ví dụ: cab thay vì chỉ nhìn thấy số 468). Tên lớp đôi khi có thể bị trùng lặp ngữ nghĩa hoặc viết khác biệt một chút, nên kết hợp cả ID và hệ thống phân loại sẽ giúp định danh chính xác lớp đó.

5. Đối với bài toán phân loại ảnh đơn nhãn (single-label classification) khi có nhiều đối tượng, guideline cho người gán nhãn cần quy định rõ các tiêu chí ưu tiên, ví dụ như:
- Diện tích/Độ nổi bật: Chọn vật thể chiếm diện tích lớn nhất, rõ nét nhất hoặc nằm ở vị trí trung tâm của bức ảnh.
- Ngữ cảnh chính (Main Subject): Đánh giá chủ đề cốt lõi mà bức ảnh muốn truyền tải (ví dụ: ảnh có cả người và bàn ăn, nhưng bối cảnh là một nhà hàng thì ưu tiên nhãn bối cảnh chung nếu taxonomy cho phép).
- Quy tắc phân cấp (Hierarchy rules): Cung cấp thứ tự ưu tiên cụ thể nếu các vật thể xuất hiện cùng nhau (ví dụ: nếu có người và xe, luôn ưu tiên gán nhãn người).

6. Không được lấy score làm ground truth (nhãn thực tế) vì score chỉ là con số xác suất dự đoán chủ quan của mô hình, không phản ánh sự thật khách quan. Một mô hình hoàn toàn có thể dự đoán sai nhưng lại có độ tự tin (score) rất cao. Ground truth bắt buộc phải là nhãn chính xác 100% do con người (hoặc quy trình chuẩn) xác định để làm thước đo đánh giá xem mô hình dự đoán đúng hay sai.

## 2\. Phát hiện vật thể
1. Chép một record: `class\_name`=person; `score`=0.912625 ; `bbox\_xyxy`=\[385.33, 69.24, 498.92, 348.92] ; `bbox\_width`=113.58 ; `bbox\_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái (top-left): Nằm ở tọa độ x = 385.33 (cách lề trái 385.33 pixel) và y = 69.24 (cách lề trên 69.24 pixel).
Góc dưới-phải (bottom-right): Nằm ở tọa độ x = 498.92 (cách lề trái 498.92 pixel) và y = 348.92 (cách lề trên 348.92 pixel).
Kích thước: Chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
threshold=0.20: 17 vật thể
threshold=0.60: 6 vật thể 

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
 Khi hạ ngưỡng độ tin cậy (ví dụ từ 0.60 xuống 0.20), mô hình sẽ đưa ra nhiều dự đoán hơn, bao gồm cả những dự đoán có độ chắc chắn thấp. Do đó, khối lượng công việc của người reviewer (người kiểm tra) sẽ tăng lên đáng kể. Cụ thể:
- Họ phải lọc và xóa bỏ nhiều hộp giới hạn bị sai (false positives - nhận diện nhầm vật thể này thành vật thể khác, hoặc khoanh vùng vào khoảng không).
- Kiểm tra kỹ lưỡng hơn các hộp bị trùng lặp (nhiều hộp cùng trỏ vào một vật).
- Điều chỉnh lại ranh giới của các hộp nếu chúng bị vẽ quá rộng hoặc quá chật so với thực tế.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Tight Bounding Box: Hộp giới hạn phải bao quanh toàn bộ phần nhìn thấy được của vật thể một cách vừa vặn nhất có thể. Không được chừa ra những khoảng trống không cần thiết giữa cạnh của hộp và ranh giới ngoài cùng của vật thể, nhưng cũng không được cắt xén mất bất kỳ phần nào của vật (trừ những phần bị che khuất tự nhiên bởi vật khác).

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- chiếc chảo/nồi màu đồng nằm treo ngay phía sau lưng người đàn ông đang bị che khuất một phần lớn (occluded) bởi cơ thể người.
- Trường hợp này nên được viết thành quy tắc trong guideline thay vì đưa lên hỏi Lab Coach. Lý do là việc vật thể chồng chéo lên nhau là một hiện tượng rất phổ biến trong ảnh thực tế (đặc biệt là ảnh có mật độ cao như nhà bếp hay giao thông). Guideline cần đưa ra nguyên tắc chung ngay từ đầu, ví dụ: "Đối với vật thể bị che khuất một phần, chỉ vẽ bounding box bao quanh những phần có thể nhìn thấy được của vật thể đó. Nếu phần bị che khuất lớn hơn 80% (tùy quy định), có thể bỏ qua không gán nhãn." Việc quy định sẵn giúp người gán nhãn tự chủ động xử lý các tình huống tương tự mà không cần chờ hướng dẫn từng trường hợp.

## 3\. Phân đoạn theo từng đối tượng
1. `instance\_id`=kitchen-001 ; `class\_name`=person; `score`=0.899318 ; số điểm polygon=`polygon\_point\_count`=348; 3–5 điểm đầu của `polygon\_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
 So với bounding box (chỉ vẽ được một hình chữ nhật bao quanh vật thể một cách thô sơ), polygon (đa giác) cung cấp thông tin chi tiết và chính xác hơn nhiều về đường viền (contour) thực tế của vật thể.
 Bằng cách nối hàng chục đến hàng trăm điểm tọa độ lại với nhau (như trong ví dụ trên là 348 điểm), polygon có thể uốn lượn bám sát theo các đường cong, góc cạnh, phần lồi lõm của đối tượng, giúp ta biết chính xác pixel nào thuộc về vật thể (foreground) và pixel nào là nền (background).

3. `instance\_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id là một chuỗi định danh duy nhất (ví dụ: kitchen-001) được gán cho một cá thể vật thể cụ thể xuất hiện trong một bức ảnh cụ thể.
- Không phải class ID: Nó không đại diện cho phân lớp của vật thể. Trong một ảnh có thể có 3 người, cả 3 đều chung class_id = 0 (person), nhưng sẽ có 3 instance_id khác nhau (ví dụ: kitchen-001, kitchen-002, kitchen-003) để phân biệt người này với người kia.

- Không phải tracking ID: Nó không dùng để theo dõi (track) vật thể đó qua nhiều khung hình (video). ID này chỉ có ý nghĩa cục bộ trong khuôn khổ của một bức ảnh tĩnh duy nhất (như được thể hiện qua tiền tố kitchen-).

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Pixel-perfect boundary: "Đường polygon phải được vẽ bám sát viền ngoài cùng của vật thể một cách chính xác nhất có thể, không được lấn ra ngoài phần nền (background) hoặc lẹm vào trong thân vật thể. Đối với các vật thể có lỗ hổng ở giữa (như tay cầm cốc, khoảng trống giữa hai chân), cần vẽ thêm các polygon phụ để khoét bỏ phần nền bên trong (nếu công cụ hỗ trợ)."

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trong ảnh kitchen, có một cái rổ/chậu cây (potted plant) ở góc trên bên trái đang bị cắt mép nghiêm trọng bởi khung hình và bị mờ nhoe nhoét.
Nên dùng guideline quy định sẵn. Tương tự như bounding box, việc vật thể bị che khuất (occlusion), bị cắt xén (truncation) hoặc mờ (blur) ở biên là rất phổ biến. Guideline phân đoạn (segmentation) cần quy định rõ ngay từ đầu, ví dụ: "Chỉ phân đoạn những phần vật thể có thể nhìn thấy rõ ràng. Nếu biên vật thể bị mờ nhòe do out-of-focus hoặc chuyển động, hãy cố gắng ước lượng đường biên ở vị trí giữa phần vật thể mờ và phần nền, hoặc vẽ theo phần điểm ảnh rõ nét nhất còn lại". Đưa quy định vào guideline giúp luồng công việc diễn ra trơn tru mà không cần tốn thời gian leo thang (escalation) cho những lỗi quá thông thường.

## 4\. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

|Tác vụ|Ground truth trông như thế nào? (đơn vị nhãn)|Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON|Annotator làm gì?|Reviewer xem gì?|
|-|-|-|-|-|
|Phân loại ảnh|1 nhãn đại diện cho toàn ảnh (class_id, class_name).|Ảnh traffic có nhiều loại xe, khó chỉ chọn 1 nhãn duy nhấtChọn 1 nhãn mô tả đúng nhất chủ đề ảnh.|Chọn 1 nhãn mô tả đúng nhất chủ đề ảnh.|Kiểm tra nhãn có khớp taxonomy và nội dung chính không.|
|Phát hiện vật thể|Hộp giới hạn (Bounding box xyxy) + nhãn.|Vật thể bị lấp/chồng chéo nhau trong ảnh kitchen.|Vẽ hộp khít phần nhìn thấy của từng vật.|Kiểm tra hộp bị rộng/chật, sai nhãn, sót vật.|
|Instance segmentation|Đa giác (Polygon nhiều điểm xy) + instance_id.|Vật (potted plant) bị cắt mép hoặc nhòe viền.|Chấm điểm sát viền vật thể để tạo mask.|Kiểm tra viền lẹm nền/lẹm vật, nhầm|

## 5\. An toàn dữ liệu

1. tôi cam kết bảo mật tuyệt đối, không sao chép, lưu trữ hay chia sẻ bất kỳ dữ liệu nào (hình ảnh, file JSON, tài liệu) của bài lab ra bên ngoài dưới mọi hình thức.
2. Nếu gặp dữ liệu hoặc ảnh không đúng phạm vi, tôi sẽ dừng xử lý và báo cáo lên cho Lab Coach để xin hướng dẫn.

## 6\. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

* \[ ] `REPORT.md` (file này đã điền)
* \[ ] `day1\_lab\_outputs/classification\_predictions.json`
* \[ ] `day1\_lab\_outputs/detection\_predictions.json`
* \[ ] `day1\_lab\_outputs/segmentation\_predictions.json`
* \[ ] `day1\_lab\_outputs/IMAGE\_ATTRIBUTION.md`
* \[ ] `day1\_lab\_outputs/visuals/classification\_top5.png`
* \[ ] `day1\_lab\_outputs/visuals/detection\_predictions.png`
* \[ ] `day1\_lab\_outputs/visuals/segmentation\_prediction.png`
* \[ ] Ô validation cuối notebook báo `PASS`
* \[ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
* \[ ] Đã nộp **link repository** trên LMS

