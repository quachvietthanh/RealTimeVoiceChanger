# Báo cáo A/B: padding phản chiếu 16 kHz trước trích xuất đặc trưng

**Đối tượng:** `RVCAdapter.convert_chunk` (`src/voice_changer/conversion/rvc/adapter.py`), dòng `np.pad(audio_16k, self.pad_16k, mode="reflect")`.
**Yêu cầu gốc:** `thuat-toan.md`. Kiểm tra giả thuyết (80, 80) làm lệch nội dung khoảng 5 ms, so sánh với (0, 160), phân tích click ở ranh giới chunk. Chưa sửa code production cho đến khi có kết luận.

## 1. Cách tái lập

```powershell
.venv\Scripts\python.exe scripts\ab_padding_test.py
```

- Kết quả JSON: `output/ab_padding/ab_padding_results.json` (gitignored; ghi kèm commit, SHA-256 của model và index).
- Tín hiệu kiểm thử: tổng hợp, 6 s, 40 kHz, hài 120 Hz có pitch glide và cổng âm tiết 0,4 s; seed `1234`, `2024`, `7`.
- Mỗi cấu hình được chạy hai cách: `whole` (một lần `convert_chunk` cho toàn bộ tín hiệu) và `chunked` (chunk 0,5 s = 20000 mẫu, ghép lại như pipeline).
- Model: `vctk_female_p231` (placeholder), `pitch_shift=10`, `f0_method=rmvpe`, `index_rate=0.5`, `filter_radius=2`, `protect_rate=0.4`.

| Ký hiệu | `pad_16k` (trái, phải) | Ý nghĩa |
| :--- | :--- | :--- |
| A | (80, 80) | Cấu hình production hiện tại |
| B | (0, 160) | Theo yêu cầu trong `thuat-toan.md` |
| C | (0, 0) | Đối chứng, không padding |

## 2. Kết quả

### 2.1 Lệch thời gian (đầu ra toàn file so với đầu vào)

Đo bằng thời điểm onset của cổng âm tiết (15 onset mỗi tín hiệu, ngưỡng 30% đỉnh envelope RMS 20 ms). Giá trị thô là hiệu giữa onset đầu ra và đầu vào, được ghép theo thứ tự. Vì chu kỳ là 400 ms nên giá trị thô xấp xỉ −400 ms nghĩa là lệch một chu kỳ, cộng thêm 400 ms để có lệch thực.

| Cấu hình | seed 1234 | seed 2024 | seed 7 | Lệch thực (trung bình) |
| :--- | ---: | ---: | ---: | ---: |
| A (80, 80) | −389,6 | −390,2 | −389,9 | **≈ +10,1 ms** |
| B (0, 160) | −398,8 | −399,3 | −399,1 | **≈ +0,9 ms** |
| C (0, 0) | −398,7 | −399,3 | (n = 14, không ghép được) | **≈ +0,7 ms** |

**Hiệu A − B = 9,2 ms** (ổn định qua ba seed). Padding trái 80 mẫu ở 16 kHz làm đầu ra trễ khoảng 9 ms. Padding phải 160 mẫu không làm lệch (B ≈ C).

Giả thuyết trong `thuat-toan.md` dự đoán khoảng 5 ms. Kết quả đo được cao hơn khoảng 1,8 lần. Nguồn gốc chênh lệch chưa được xác định.

### 2.2 Click ở ranh giới chunk

Chỉ số: độ nhảy mẫu tại ranh giới chia cho median độ nhảy trong cửa sổ ±480 mẫu. Chỉ tính các ranh giới có năng lượng (không im lặng). Giá trị càng gần 1 càng tốt.

| Cấu hình | Trung bình qua 3 seed | Ghi chú |
| :--- | ---: | :--- |
| A (80, 80) | ~29 (21–37) | Có click rõ |
| B (0, 160) | ~29 (18–50) | Không khác A |
| C (0, 0) | 17 000–93 000 | Nhảy do padding edge (xem 2.3) |

Dạng sóng seed 1234 (đoạn 1,5 s, ranh giới tại mẫu 20000):

| | Độ nhảy tại ranh giới | Đầu ra toàn file cùng vị trí |
| :--- | ---: | ---: |
| A | **0,442** | 0,012 |
| B | **0,353** | 0,023 |

Đầu ra toàn file rất mượt. Đầu ra chunk bị nhảy đột ngột: mẫu cuối chunk 1 ≈ 0,03, mẫu đầu chunk 2 ≈ 0,47 (cấu hình A).

Khoảng cách phổ log-STFT (LSD) gần ranh giới ≈ 10 dB, xa ranh giới ≈ 8 dB, không khác biệt giữa A và B. Chỉ số này không phân biệt được hai cấu hình.

### 2.3 Cấu hình C: padding edge

Với (0, 0), đuôi chunk đầu ra là một giá trị phẳng lặp lại (0,4454 cho ít nhất 5 mẫu cuối). Đây là hệ quả của nhánh `np.pad(output_audio, (0, pad_needed), mode="edge")` trong `adapter.py` (dòng ~277), được kích hoạt khi đầu ra ngắn hơn `expected_len`. Điều này cho thấy padding không chỉ là chi tiết kỹ thuật: nó giữ độ dài đầu ra đúng. Chưa kiểm tra xem nhánh edge có được kích hoạt ở A và B hay không.

## 3. Kết luận

1. **Giả thuyết lệch thời gian: được hỗ trợ một phần.** Padding trái (80 mẫu) làm đầu ra trễ khoảng 9 ms so với (0, 160), hướng đúng như giả thuyết nhưng biên độ lớn hơn ước tính 5 ms.
2. **Giả thuyết click do padding: không được hỗ trợ.** Khi đổi từ A sang B, click ở ranh giới gần như không đổi (cả hai đều khoảng 0,35–0,44 so với đầu ra toàn file 0,01–0,02). Nguyên nhân chính là chunk được suy luận độc lập, đầu ra chunk không khớp đầu ra toàn file tại cùng thời điểm (không phải lệch pha đơn thuần, vì tương quan theo dịch thời gian không khớp, NRMSE > 1).
3. **Có một lỗi độc lập cần kiểm tra:** nhánh `mode="edge"` khi đầu ra ngắn hơn `expected_len` tạo đuôi phẳng. Với cấu hình production, cần xác nhận nhánh này có bị kích hoạt không.

**Không có phương án sửa nào được chứng minh đủ cơ sở.** Đổi sang (0, 160) không cải thiện click và chỉ bỏ được phần lệch 9 ms. Tôi chưa đổi code production.

## 4. Phương án đề xuất (chưa triển khai, cần xác nhận)

- **Chunk có ngữ cảnh chồng lấp:** đưa thêm 0,1–0,2 s đầu vào trước mỗi chunk, rồi bỏ phần đầu ra tương ứng với ngữ cảnh đó. Đây là cách xử lý tiêu chuẩn cho bài toán nhiều chunk độc lập. Chi phí: tăng khoảng 20–40% tính toán mỗi chunk, cần đo lại RTF và độ trễ. Tôi chưa chứng minh được cách này trên đây, cần thử nghiệm trước.
- **Bù lệch trái:** nếu giữ (80, 80), có thể bỏ đúng phần lệch đo được (≈ 9 ms, tức khoảng 360 mẫu tại 40 kHz). Chỉ nên làm sau khi giải thích được nguồn gốc chênh lệch 9 ms so với 5 ms.

## 5. Hạn chế

- Tín hiệu là tổng hợp, không phải giọng nói. Kết quả click có thể khác với tiếng nói thật.
- Chỉ dùng model placeholder `vctk_female_p231` (giọng tiếng Anh).
- Lệch thời gian đo bằng onset ở độ phân giải 1 ms. Một số lần chạy có chênh lệch lệch chu kỳ, phải hiệu chỉnh bằng cách cộng 400 ms. Hiệu A − B không bị ảnh hưởng vì cùng một hiệu chỉnh.
- Chạy trên CPU (máy không có GPU NVIDIA), tuy kết quả không phụ thuộc thiết bị về mặt thuật toán.
- Chưa kiểm tra xem nhánh edge có được kích hoạt ở A và B.

## 6. Thay đổi mã trong lần thử nghiệm này

- `src/voice_changer/conversion/rvc/adapter.py`: thêm thuộc tính `self.pad_16k = (80, 80)`, mặc định giữ nguyên hành vi cũ. Đây là điểm kiểm soát để harness thay đổi padding, không đổi hành vi production.
- `scripts/ab_padding_test.py`: harness A/B (mới).
- `docs/ab-padding-report.md`: báo cáo này.

## 7. Bổ sung: kiểm tra nhánh edge và overlap

### 7.1 Nhánh `mode="edge"` có kích hoạt không (`scripts/diagnose_edge_pad.py`)

| Cấu hình | Chunk kích hoạt edge | Độ dài thô (Applio) | Độ dài yêu cầu | Mẫu được lấp |
| :--- | :---: | ---: | ---: | ---: |
| (80, 80) production | 0/12 | 20000 | 20000 | 0 |
| (0, 160) | 0/12 | 20000 | 20000 | 0 |
| (0, 0) đối chứng | **12/12** | 19200 | 20000 | 800 (20 ms) |

Đuôi phẳng chỉ xuất hiện ở cấu hình (0, 0). Cấu hình production không bị ảnh hưởng. Kết luận mục 2.3 đúng cho đối chứng, không áp dụng cho production.

### 7.2 Bước 2 bị chặn

`tests/fixtures/input_male.wav` không có trong repo và không tìm thấy trên máy. Máy không có GPU NVIDIA (`torch.cuda.is_available()` là `False`), nên không thể chạy đúng "GPU hiện tại". Chưa có kết quả trên giọng thật.

### 7.3 Overlap ngữ cảnh trái (`scripts/overlap_test.py`, tín hiệu tổng hợp, CPU, pad (80, 80), 1 seed)

| Cấu hình | Nhảy TB tại ranh giới | Tỉ lệ so với tham chiếu | LSD gần ranh giới (dB) | Inference (s) | Chênh lệch thời gian |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0,161 | 41,4 | 10,69 | 8,49 | — |
| Overlap 100 ms | 0,188 | 19,4 (baseline 25,6 khi chạy riêng) | 10,47 | 7,12 | −9,6% |
| Overlap 200 ms | 0,105 | 24,6 | 9,55 | 10,59 | +24,7% |

Tham chiếu toàn file có nhảy TB ≈ 0,008.

- Overlap 100 ms không đủ để giảm click. Dù tỉ lệ có giảm, mức nhảy thực tế gần như không đổi.
- Overlap 200 ms giảm nhảy ở ranh giới khoảng 35% và cải thiện LSD gần ranh giới, nhưng vẫn cao gấp khoảng 12 lần tham chiếu, nên chưa loại bỏ được gián đoạn.
- Số liệu thời gian chưa đáng tin: baseline luôn chạy trước nên có thể bị ảnh hưởng bởi khởi động, và mỗi cấu hình chỉ chạy một lần. Overhead đo được (−9,6% và +24,7%) không nhất quán, cần đo lại với nhiều lần lặp và thứ tự xen kẽ.
- Lượng âm thanh xử lý tăng theo lý thuyết: 100 ms là +20% (500 → 600 ms), 200 ms là +40%.

### 7.4 A/B trên giọng nam thật `tests/fixtures/input_male.wav` (CPU, 32,2 s, 1 lần chạy)

Lệnh: `.venv\Scripts\python.exe scripts\ab_padding_test.py --input tests/fixtures/input_male.wav --out output/ab_padding/ab_padding_speech.json`. Kết quả: `output/ab_padding/ab_padding_speech.json`.

| Cấu hình | Lệch envelope (ms) | Tương quan envelope | Tỉ lệ nhảy ở ranh giới (TB / max) | LSD gần ranh giới (dB) | LSD xa ranh giới (dB) | Thời gian convert (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| A (80, 80) production | **+5** | 0,67 | **39 / 433** | **9,7** | 8,1 | 59,6 |
| B (0, 160) | −1 | 0,67 | 118 / 2374 | 10,5 | 8,1 | 56,7 |
| C (0, 0) đối chứng | −1 | 0,66 | 163 813 / 4,9·10⁶ | 10,8 | 8,1 | 52,8 |

Độ dài đầu ra khớp đầu vào ở cả ba cấu hình (1 289 387 mẫu).

- **Lệch thời gian:** trên giọng thật, (80, 80) làm đầu ra trễ khoảng 5 ms so với (0, 160). Khớp với giả thuyết trong `thuat-toan.md`, và xác nhận kết quả trên tín hiệu tổng hợp (9 ms, thiên lệch do phép đo).
- **Click:** trên giọng thật, (0, 160) **tệ hơn** (80, 80) ở ranh giới: tỉ lệ TB 118 so với 39. Trên tín hiệu tổng hợp hai cấu hình gần như tương đương. Vậy trên giọng thật không nên chuyển sang (0, 160).
- Cấu hình production (80, 80) có LSD gần ranh giới thấp nhất trong ba cấu hình.
- Click vẫn còn ở cả ba cấu hình. Cấu hình production vẫn có mức nhảy trung bình gấp khoảng 39 lần đầu ra toàn file.
- Đánh đổi: giảm trễ 5 ms bằng cách bỏ padding trái sẽ làm click tệ hơn. Không có cấu hình padding nào trong ba cấu hình này giải quyết được cả hai vấn đề.
- Thời gian trên CPU: khoảng 1,85 lần thời gian thực cho 32 s âm thanh, không đạt thời gian thực.

### 7.5 Đề xuất tiếp theo (chưa thực hiện)

- Đo lại thời gian với nhiều lần lặp, xen kẽ thứ tự cấu hình, để có chi phí thật.
- Thử crossfade trong vùng overlap thay vì bỏ hẳn đầu ra ngữ cảnh. Việc này không cần thêm inference so với overlap 200 ms.
- Chạy lại toàn bộ trên `tests/fixtures/input_male.wav` khi có file và GPU.
