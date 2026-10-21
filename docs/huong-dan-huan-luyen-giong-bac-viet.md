# Hướng dẫn huấn luyện model giọng nữ Bắc Việt hiệu quả

Tài liệu này hướng dẫn cách chuẩn bị dữ liệu, thu âm, kiểm tra và chọn checkpoint để có một model RVC v2 giọng nữ tiếng Việt Bắc tự nhiên, rõ thanh điệu.

**Phạm vi:** tài liệu này bổ sung cho [custom-voice-training-plan.md](custom-voice-training-plan.md). Các lệnh huấn luyện (preprocess, extract, train, extract_index) và ràng buộc phần cứng (GPU 4 GB) được mô tả ở file đó, không lặp lại ở đây.

---

## 1. Nguyên tắc cốt lõi

Chất lượng model phụ thuộc theo thứ tự sau:

1. **Độ sạch và nhất quán của dữ liệu** quan trọng hơn số giờ thu.
2. **Độ phủ thanh điệu và âm tiết** quan trọng hơn độ dài. Tiếng Việt có 6 thanh, nên dữ liệu phải có đủ 6 thanh trên nhiều âm khác nhau.
3. **Chọn checkpoint bằng nghe thử** quan trọng hơn chọn theo số epoch hay số liệu loss.

> Một model 25 phút thu sạch, đủ thanh điệu thường tốt hơn model 2 giờ thu ồn, lẫn nhạc nền hoặc có nhiều giọng khác nhau.

---

## 2. Yêu cầu về người nói và sự đồng thuận

- Chỉ dùng **một người nói duy nhất**, là nữ trưởng thành, có **sự đồng ý bằng văn bản** cho việc huấn luyện và sử dụng giọng nói trong dự án này.
- Giữ nguyên tắc của mục 2.1 trong [custom-voice-training-plan.md](custom-voice-training-plan.md): không dùng cho giả mạo, không phân phối lại danh tính giọng nói.
- Không trộn dữ liệu từ VIVOS, ViVoice hay Common Voice nếu chưa xác minh được người nói và quyền sử dụng.

### Về giọng Bắc

Giọng "chuẩn" Bắc không đồng nhất giữa các người nói. Cách làm đúng là ghi lại cách phát âm **tự nhiên của người nói**, không ép theo một chuẩn sách vở. Khi kiểm tra, các điểm sau thường giúp nhận ra đặc trưng Bắc:

- Phân biệt rõ **tr / ch** và **s / x** (giọng Nam thường gộp).
- Thanh **hỏi** và **ngã** được phát âm khác nhau. Đây là điểm dễ thấy nhất khi nghe.
- Các cặp **l / n** và **v / d / gi** cần kiểm tra theo cách người nói thật sự phát âm, đừng giả định.

---

## 3. Chuẩn bị phòng và thiết bị thu

| Hạng mục | Khuyến nghị |
| :--- | :--- |
| Phòng | Phòng nhỏ, nhiều vật liệu mềm (rèm, thảm, sofa). Tránh phòng vang, tiếng quạt, điều hòa, tiếng xe. |
| Micro | Micro condenser hoặc dynamic chất lượng tốt. Đặt cố định cách miệng khoảng **15–20 cm**, hơi lệch khỏi trục miệng để giảm tiếng "p" và "b" bật hơi. |
| Pop filter | Bắt buộc, để giảm tiếng xì và hơi thở mạnh. |
| Định dạng | WAV PCM **24-bit, 48 kHz**, mono. |
| Mức thu | Trung bình khoảng **-18 dBFS**. Đỉnh không vượt **-6 dBFS**. Tuyệt đối không để clipping. |
| Tư thế | Đứng hoặc ngồi thẳng, không cúi. Giữ vị trí micro và mức gain cố định trong toàn bộ phiên. |

**Thử trước khi thu:** thu 30 giây thử, nghe lại bằng tai nghe kín. Nếu có tiếng ù hoặc tiếng vang, dừng và xử lý phòng trước.

---

## 4. Kịch bản thu âm

Tổng thời gian thu nên khoảng **40–60 phút** ghi âm, sau khi loại khoảng lặng và bản thu hỏng thì còn khoảng **30–45 phút** dữ liệu sạch. Tối thiểu chấp nhận được là 20 phút sạch.

Kịch bản gồm bốn nhóm câu. Mỗi nhóm nên chiếm tỉ lệ tương ứng trong bản thu.

### 4.1. Nhóm A: cặp thanh điệu (khoảng 20%)

Đọc nhiều lần cùng một âm với 6 thanh. Đây là nhóm quan trọng nhất cho việc giữ thanh điệu sau khi chuyển đổi.

- ma, má, mà, mả, mã, mạ
- ba, bá, bà, bả, bã, bạ
- lê, lề, lệ, lể, lễ, lẹ (thay "lệ" thứ hai để có đủ 6 thanh)
- ca, cá, cà, cả, cã, cạ
- Ví dụ câu: "Ma là má của bà." (đọc tự nhiên, không đọc từng từ rời rạc)

### 4.2. Nhóm B: phụ âm cuối và âm tiết kết thúc (khoảng 20%)

Các phụ âm cuối giúp phân biệt âm tiết và rất dễ bị mờ khi chuyển đổi.

- **-p, -t, -c, -ch** (tắc): tháp, sát, bác, mạch, cắt, nhịp
- **-m, -n, -ng, -nh** (vang mũi): lâm, hàn, làng, lãnh, cổng, mình
- Câu mẫu: "Cô ấy vừa đi về nhà lúc năm giờ."

### 4.3. Nhóm C: câu đàm thoại tự nhiên (khoảng 40%)

Đây là phần mô phỏng giọng nói thực tế khi chat và gọi thoại. Viết câu đa dạng về độ dài, cảm xúc và loại câu.

- Câu kể: "Hôm nay trời đẹp quá, chúng ta đi dạo một vòng nhé."
- Câu hỏi: "Bạn đã ăn cơm chưa?"
- Câu cảm thán nhẹ: "Ồ, hay thật đấy!"
- Câu dài nhiều mệnh đề: "Tôi nghĩ nếu mình đi sớm thì sẽ kịp chuyến xe buổi sáng, còn không thì phải đợi đến chiều."
- Câu thân mật, xưng hô: "Anh ơi, em gửi tài liệu rồi nhé."

### 4.4. Nhóm D: âm Bắc và từ dễ nhầm (khoảng 20%)

Đọc các cặp từ để model học được sự khác biệt thật của người nói.

- tr / ch: trăng – chăng, trả – chả, trời – chời
- s / x: sa – xa, sắn – xắn, sôi – xôi
- Số và đơn vị: "mười lăm", "hai mươi mốt", "năm trăm nghìn", "tám giờ rưỡi"
- Từ ghép và từ thường dùng: "máy tính", "điện thoại", "cảm ơn", "xin lỗi", "không sao đâu"

> Ghi chú: các cặp trên là gợi ý. Nếu người nói phát âm khác (ví dụ không phân biệt được một cặp), hãy để nguyên cách họ nói, đừng ép phát âm.

### 4.5. Quy tắc khi đọc

- Đọc tự nhiên, như đang nói chuyện với người quen. Không đọc quá rõ hoặc giả giọng.
- Giữ cùng một khoảng cách micro và cùng mức âm lượng.
- Nghỉ giữa các câu. Mỗi file nên chứa 1–3 câu, dài khoảng 3–10 giây.
- Tránh cười, thì thầm, hát, hoặc nói lớn tiếng nếu không có chủ ý. Những đoạn này làm model bị lệch.
- Nghỉ 5–10 phút mỗi 20–30 phút để tránh mệt giọng. Giọng mệt làm đổi âm sắc.

---

## 5. Tổ chức thư mục và kiểm tra dữ liệu

### 5.1. Cấu trúc đề xuất

```text
data/
├── dataset_raw/            # File gốc đã thu, đặt tên theo phiên: session01_0001.wav
├── held_out_test/          # 1–2 phút KHÔNG dùng để huấn luyện (xem mục 6.2)
└── transcripts/            # Văn bản đọc của từng file (tùy chọn, hữu ích khi kiểm tra)
```

Đặt tên file theo thứ tự có ý nghĩa, ví dụ `session02_0042.wav`, để dễ truy vết khi có file lỗi.

### 5.2. Kiểm tra trước khi huấn luyện

Chạy kiểm tra dữ liệu theo lệnh trong [custom-voice-training-plan.md](custom-voice-training-plan.md) (Step 1). Đảm bảo:

- [ ] Không có clipping, không có file lỗi giải mã.
- [ ] Không có file nào chứa tiếng người khác, tiếng nhạc hay tiếng ồn lớn.
- [ ] Mức âm lượng trung bình khác nhau không quá 3 dB giữa các phiên thu.
- [ ] Tổng thời gian sạch đạt mục tiêu (xem mục 4).
- [ ] Đủ 6 thanh điệu trên nhiều âm, có đủ các phụ âm cuối.

### 5.3. Làm sạch

- **Không** dùng bộ lọc khử ồn quá mạnh. Lọc mạnh làm mất hơi thở và phụ âm xát (s, x, ph, kh), dễ tạo tiếng rè.
- Chỉ dùng cắt khoảng lặng đầu và cuối. Giữ lại khoảng lặng ngắn bên trong câu.
- Chuẩn hóa âm lượng về cùng mức trung bình, đừng nén dynamic quá mức.
- Không dùng EQ hay hiệu ứng lên dữ liệu gốc. Để model học giọng thật.

---

## 6. Huấn luyện và chọn checkpoint

### 6.1. Thứ tự thực hiện

1. Chạy smoke test theo bước **Step 4A** của [custom-voice-training-plan.md](custom-voice-training-plan.md). Xác nhận không lỗi và không tràn VRAM.
2. Chạy huấn luyện chính theo bước **Step 4B** của file đó. Nên lưu checkpoint mỗi **25 epoch**.
3. Từ mỗi checkpoint, xuất model inference và chạy đánh giá theo mục 7.
4. Dừng khi chất lượng nghe thử không còn cải thiện, không dựa riêng vào số epoch.

### 6.2. Tập held-out

- Giữ 1–2 phút cho tập held-out **trước** khi tiền xử lý. Tập này không được xuất hiện trong huấn luyện.
- Dùng tập này để nghe đánh giá, vì nó là câu người nói đã đọc nhưng model chưa từng học.
- Nếu tập held-out lọt vào dữ liệu huấn luyện, kết quả đánh giá không còn đáng tin.

### 6.3. Số epoch

- Với fine-tune từ pretrained RVC v2, chất lượng thường đạt đỉnh khoảng **100–200 epoch**. Đây là điểm tham khảo, không phải đích cố định.
- Vượt quá điểm đỉnh thường làm giọng bị kim loại, khô, hoặc robot. Nếu nghe thấy dấu hiệu này, lùi về checkpoint trước đó.

---

## 7. Đánh giá chất lượng

### 7.1. Bộ câu kiểm tra cố định

Chuyển đổi cùng một bộ câu qua mọi checkpoint để so sánh công bằng:

1. **Nhóm thanh điệu:** "ma má mà mả mã mạ" và "ba bá bà bả bã bạ". Đánh giá xem 6 thanh có phân biệt được không.
2. **Nhóm phụ âm cuối:** "tháp sát bác mạch", "lâm hàn làng lãnh". Đánh giá xem phụ âm có rõ không.
3. **Câu đàm thoại:** lấy 2–3 câu từ held-out, và chuyển đổi `tests/fixtures/input_male.wav` để kiểm tra giọng nam chuyển sang giọng nữ.

Chạy chuyển đổi với profile tiếng Việt đã có trong dự án:

```powershell
python scripts/offline_conversion_test.py --model models/weights/<model>.pth --index models/indices/<model>.index --input tests/fixtures/input_male.wav --language vi
```

### 7.2. Thang điểm nghe thử

Chấm mỗi checkpoint từ 1 đến 5 trên các tiêu chí sau. Ghi lại kèm tên checkpoint và ngày nghe.

| Tiêu chí | 1 (kém) | 5 (tốt) |
| :--- | :--- | :--- |
| Độ rõ lời | Khó hiểu, nuốt âm | Mọi âm rõ, dễ hiểu |
| Thanh điệu | Các thanh lẫn vào nhau | 6 thanh phân biệt rõ |
| Phụ âm cuối | Mờ, mất âm | Rõ, không bị cắt |
| Độ tự nhiên | Robot, lặp lại | Giống người thật |
| Độ ấm giọng | Kim loại, gắt | Ấm, mềm, dễ chịu |
| Tạp âm | Rè, lạo xạo, hụt hơi | Sạch |

Chọn checkpoint có tổng điểm cao nhất. Nếu hai checkpoint gần bằng nhau, chọn checkpoint sớm hơn vì ít rủi ro over-train hơn.

### 7.3. Chỉnh tham số chuyển đổi sau huấn luyện

Sau khi có model, tinh chỉnh `index_rate` cho model mới. Profile `vi` của dự án đặt `index_rate = 0.50`, nhưng tối ưu thật sự phụ thuộc vào model:

- Nếu giọng nghe "không giống người nói", tăng `index_rate` lên 0.6–0.7.
- Nếu giọng bị gắt, kim loại, giảm `index_rate` xuống 0.3–0.4.
- Nếu thanh điệu bị sai, kiểm tra lại dữ liệu nhóm A trước khi chỉnh tham số.

---

## 8. Xử lý lỗi thường gặp

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
| :--- | :--- | :--- |
| Giọng kim loại, sắc | Over-train; `index_rate` quá cao | Lùi checkpoint; giảm `index_rate` |
| Thanh điệu bị lẫn | Thiếu dữ liệu nhóm A; `filter_radius` quá lớn | Thu bổ sung nhóm A; dùng profile `vi` |
| Phụ âm cuối bị mờ | Thiếu dữ liệu nhóm B; `protect_rate` thấp | Thu bổ sung nhóm B; tăng `protect_rate` |
| Có tiếng rè, lạo xạo | Khử ồn quá mạnh; clipping | Thu lại với mức thấp hơn; bỏ bước lọc mạnh |
| Giọng không giống người nói | Dữ liệu ít; `index_rate` thấp | Thu thêm; tăng `index_rate` |
| Giọng lẫn tiếng người khác | Dữ liệu nhiều người | Loại file đó, huấn luyện lại |
| Hết VRAM khi huấn luyện | `batch_size` quá lớn; `cache_data_in_gpu = True` | Dùng `batch_size = 1`; đảm bảo cache trong GPU là `False` |

---

## 9. Danh sách kiểm tra trước khi triển khai

- [ ] Có sự đồng ý bằng văn bản của người nói.
- [ ] Dữ liệu chỉ có một người nói.
- [ ] Đủ 6 thanh điệu và đủ các nhóm phụ âm cuối.
- [ ] Held-out được tách trước khi tiền xử lý.
- [ ] Checkpoint được chọn bằng nghe thử theo thang điểm mục 7.2.
- [ ] Đã thử cả hai chế độ `--language vi` và `--language en` để so sánh.
- [ ] Model và index được lưu vào `models/weights/` và `models/indices/`, không commit vào Git (đã có trong `.gitignore`).
- [ ] Ghi lại tham số đã dùng (số epoch, `index_rate`, `protect_rate`, `filter_radius`) vào `docs/selected-voice-model.md` hoặc file tương tự.
