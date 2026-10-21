# Hướng dẫn Ứng dụng Desktop GUI — RealTimeVoiceChanger

Tài liệu hướng dẫn vận hành, kiến trúc, kiểm định và bố cục giao diện đồ họa người dùng (Desktop GUI) của dự án **RealTimeVoiceChanger**.

---

## 1. Giới thiệu & Khởi chạy

Desktop GUI được xây dựng hoàn toàn bằng ngôn ngữ Python với thư viện chuẩn `tkinter` và `ttk`, áp dụng bảng màu Dark Mode hiện đại, độ tương phản cao, hỗ trợ hiển thị tối ưu trên Windows (DPI-aware: 100%, 125%, 150%), kết nối trực tiếp với backend service thực (`VoiceChangerService`) theo đúng hợp đồng `docs/gui-backend-contract.md`.

### Cách khởi chạy ứng dụng:

Chạy ứng dụng từ thư mục gốc của repository với môi trường ảo `.venv`:

```powershell
# Cách 1: Sử dụng launcher script chuyên dụng
.venv\Scripts\python scripts/run_gui.py

# Cách 2: Sử dụng entry point chính của package
.venv\Scripts\python -m voice_changer.main --gui

# Hoặc nếu package đã được cài đặt ở chế độ editable (pip install -e .)
voice-changer
```

*(Lưu ý: Thêm tham số `--cli` nếu muốn chạy giao diện dòng lệnh: `voice-changer --cli`).*

---

## 2. Thiết kế Giao diện & Trải nghiệm Người dùng (Visual Design)

Phiên bản GUI được thiết kế lại toàn diện để mang lại trải nghiệm chuyên nghiệp, thoáng đãng, dễ quét mắt (scannable) và không bị cắt chữ khi co giãn:

### 2.1. Hệ màu & Kiểu chữ (Theme & Typography)
- **Bảng màu Restrained Dark Mode**:
  * Nền chính (`BG_DARK`): `#0d131f`
  * Khối nội dung / thẻ (`BG_SURFACE`): `#161f2e`
  * Phần tử tương tác / thẻ nổi (`BG_ELEVATED`): `#1f2b3e`
  * Màu nhấn chủ đạo (`PRIMARY`): `#3b82f6` (Blue hiện đại, tương phản chuẩn WCAG AA)
  * Màu trạng thái tích cực (`SUCCESS`): `#10b981` (Emerald green)
  * Màu cảnh báo (`WARNING`): `#f59e0b` (Amber)
  * Màu lỗi / dừng (`DANGER`): `#f43f5e` (Rose red)
- **Typography chuẩn Windows**:
  * Sử dụng font hệ thống `Segoe UI` với kích cỡ từ 9.5pt (chú thích) đến 16pt (tiêu đề chính).
  * Độ tương phản chữ vượt trội: Văn bản chính `#f8fafc` (trắng sáng), văn bản phụ `#cbd5e1` (xám nhạt), chú thích `#94a3b8`.
  * Loại bỏ hoàn toàn các khối chữ li ti, dày đặc khó đọc.

### 2.2. Xử lý DPI & Co giãn kích thước cửa sổ (Responsive Layout)
- **Kích thước khuyến nghị**: `1080 x 760` (mặc định), kích thước tối thiểu an toàn: `960 x 680`.
- **Hỗ trợ Windows DPI**: Gọi trực tiếp `shcore.SetProcessDpiAwareness(1)` trên Windows để các font chữ và thanh trượt hiển thị sắc nét ở mọi mức scaling (100%, 125%, 150%).
- **Chống tràn & cắt chữ (Dynamic Label Wrapping)**: Các nhãn mô tả dài sử dụng sự kiện `<Configure>` động (`event.width - padding`) thay vì cố định `wraplength`, tự động dàn dòng mượt mà khi người dùng phóng to hoặc thay đổi kích thước cửa sổ.
- **Thanh cuộn thông minh**: Cả hai tab (Dashboard và Hướng dẫn kết nối) đều được bọc trong `Canvas` cuộn tự động chỉ khi nội dung vượt quá chiều cao cửa sổ.

---

## 3. Bố cục & Các khối chức năng chính (Main Dashboard)

Giao diện tổ chức theo quy trình thao tác tự nhiên 5 bước của người dùng:
1. **Chọn hồ sơ giọng nói**.
2. **Chọn thiết bị Micro & Loa / Cáp ảo**.
3. **Cân chỉnh cao độ & tham số suy luận**.
4. **Bắt đầu / Dừng chuyển đổi**.
5. **Theo dõi độ trễ & độ ổn định thời gian thực**.

### Khối 1: Hồ sơ giọng nói (Voice Profiles)
- Danh sách thả xuống rõ ràng, tự động chọn profile mặc định (`vctk_p231`).
- **Huy hiệu thẩm định chất lượng duy nhất & chuẩn xác**:
  * `ℹ️ BẢN MẪU KỸ THUẬT`: Dành cho `vctk_p231` (Tiếng Anh, checkpoint mẫu dùng để nghiệm thu pipeline).
  * `⚠️ THỬ NGHIỆM (CHỜ DUYỆT)`: Dành cho `voice_a_preview` (Người nói A, e50) và `voice_b_preview` (Người nói B, e25). Không tự động thăng hạng hay gán nhãn chính thức khi chưa được Project Owner phê duyệt.
  * `✓ ĐÃ PHÊ DUYỆT`: Chỉ hiển thị khi trạng thái là `approved`.
- **Tình trạng tài nguyên**: Hiển thị rõ ràng trạng thái tệp checkpoint `.pth` và tệp chỉ mục `.index`.
- Nút bấm **"Nạp mô hình vào GPU"** (`Primary.TButton`) kèm phản hồi trực quan khi đang tải VRAM.

### Khối 2: Thiết bị âm thanh (Audio Devices)
- Phân định rõ ràng:
  * **Microphone đầu vào**: Nhận dạng tên thiết bị và chuẩn driver (`[MME]`, `[DirectSound]`, `[WASAPI]`).
  * **Thiết bị phát âm thanh đầu ra**: Tự động nhận diện cáp ảo và đánh dấu nổi bật `⭐ [Cáp ảo]` đối với các thiết bị như `CABLE Input`.
- **Cảnh báo tương thích WASAPI**: Hiển thị cảnh báo trực quan ngắn gọn khi phát hiện thiết bị dùng WASAPI (nguy cơ lỗi tần số mẫu 40 kHz với PortAudio).
- Tự động ghi nhớ thiết bị đã chọn theo cặp định danh `(device_name, host_api)` tại `%LOCALAPPDATA%\RealTimeVoiceChanger\gui_settings.json`.
- Nút **"🔄 Làm mới"** danh sách thiết bị khi người dùng cắm thêm tai nghe hoặc micro mới.

### Khối 3: Khử tiếng ồn microphone (Microphone Noise Suppression)
- Nằm ngay dưới khối Thiết bị âm thanh, xử lý tín hiệu micro đầu vào **trước khi đưa vào chuyển đổi giọng nói RVC**.
- **Điều khiển BẬT/TẮT rõ ràng**:
  * **Mặc định là TẮT (OFF)** theo đúng mặc định của backend. Không tự động kích hoạt khi mở ứng dụng.
  * Hiển thị trạng thái backend thực tế: `● ĐANG TẮT`, `● ĐÃ BẬT (CHỜ STREAMING)`, `● ĐANG LỌC TIẾNG ỒN`, hoặc `⚠️ BỘ LỌC TỰ NGẮT DO LỖI`.
- **3 chế độ khử ồn chuẩn backend (Wiener Gain Floor)**:
  * `Nhẹ (-6 dB)` (`light`): Phòng yên tĩnh, bảo toàn tối đa các âm gió và giọng thì thào.
  * `Vừa (-12 dB)` (`medium` - Chế độ chuẩn): Phù hợp tiếng quạt gió máy tính hoặc tiếng ù nền thông thường.
  * `Mạnh (-18 dB)` (`strong`): Dành cho môi trường ồn lớn hoặc quạt tản nhiệt mạnh.
- **Độ trễ & Hiệu năng thực tế**:
  * Thêm độ trễ cố định `+20.0 ms` (1 frame STFT 20 ms ở 40 kHz), giữ nguyên dù BẬT hay TẮT để không làm trễ nhịp âm thanh.
  * Hoạt động 100% trên CPU (khoảng 0.3–0.5 ms mỗi chunk 500 ms), không tiêu tốn tài nguyên GPU VRAM.
- **Cảnh báo âm sắc**: Cảnh báo người dùng rằng mức khử ồn quá mạnh có thể ảnh hưởng đến phụ âm vô thanh, âm gió và tiếng thì thào tự nhiên.
- **Xử lý sự cố & Tự phục hồi**: Nếu gặp 3 lỗi liên tiếp trong phiên streaming, bộ lọc tự ngắt (`failed=True`) để không làm hỏng âm thanh đầu ra; người dùng có thể nhấn nút **"🔄 Kích hoạt lại bộ lọc"** bất kỳ lúc nào để khởi động lại.

### Khối 4: Tinh chỉnh giọng (Live Conversion Tuning)
- Cung cấp 4 thanh trượt được căn chỉnh rộng rãi, giá trị số hiển thị nổi bật, độ tương phản cao:
  * **Độ dịch cao độ (Pitch Shift)**: Từ `-24` đến `+24` bán cung (hiển thị kèm dấu `+` và đơn vị `bán cung`, khuyên dùng Nam ➔ Nữ: `+12` đến `+16`).
  * **Tỷ lệ Index (Feature Retrieval)**: Từ `0.00` đến `1.00` (bước nhảy `0.05`, pha trộn đặc trưng người nói).
  * **Bán kính lọc trung vị (Median Filter)**: Từ `0` đến `7` (bước nhảy `1`, làm mượt cao độ, giảm run giọng).
  * **Bảo vệ âm thanh / Hơi thở (Protect)**: Từ `0.00` đến `0.50` (bảo vệ phụ âm vô thanh và tiếng thở tự nhiên).
- **Áp dụng tức thì**: Kéo thanh trượt khi đang streaming sẽ áp dụng ngay ở chunk tiếp theo (độ trễ <= 0.5s) mà không cần khởi động lại luồng âm thanh.
- Nút **"↩️ Mặc định"**: Khôi phục nhanh về cấu hình khuyến nghị của Profile đang chọn.
- Chú thích thuật toán nhận diện cao độ: Cố định `RMVPE (Mặc định chuẩn dự án)`.

### Khối 4: Trạng thái & Đo lường thời gian thực (Controls & Real-Time Metrics)
- **Huy hiệu trạng thái có chấm màu (Status Dot)**:
  * `● CHƯA NẠP MODEL` (Màu xám)
  * `● ĐANG NẠP GPU...` (Màu hổ phách)
  * `● SẴN SÀNG` (Màu xanh ngọc)
  * `● ĐANG CHẠY (STREAMING)` (Màu xanh lá rực rỡ)
  * `● LỖI HỆ THỐNG` (Màu đỏ tươi)
- **Nút hành động chính (Start/Stop)**: Nổi bật, cân đối, kích thước vừa phải, chuyển đổi mượt mà giữa `▶ BẮT ĐẦU CHUYỂN ĐỔI` (`Success.TButton`) và `⏹ DỪNG CHUYỂN ĐỔI` (`Danger.TButton`).
- **Lưới thẻ chỉ số hiệu năng (6 Metric Cards)**:
  1. **RTF (Real-Time Factor)**: Tỷ lệ suy luận thời gian thực (đổi màu tự động: Xanh `< 0.85`, Vàng `0.85–1.00`, Đỏ `> 1.00`).
  2. **GPU Suy luận**: Thời gian inference trung bình trên GPU (ms).
  3. **Độ trễ phần mềm**: Ước tính độ trễ xử lý âm thanh đầu cuối (Chunk + Inference + Output Buffer).
  4. **Buffer phát lại**: Dung lượng buffer âm thanh sẵn sàng phát (ms).
  5. **Chunks (Xử lý / Rơi)**: Số lượng chunk đã hoàn thành / số chunk bị loại bỏ.
  6. **Gián đoạn âm**: Đếm số lần âm thanh bị đứt đoạn (Underflows - mục tiêu luôn là `0`).
- **Hộp thoại lỗi thông minh**: Khi xảy ra sự cố (ngắt kết nối micro, thiết bị bận), hiển thị thông điệp tiếng Việt thân thiện, gợi ý xử lý và nút "Xem chi tiết kỹ thuật" để tra cứu stack trace.

---

## 4. Hướng dẫn kết nối Cáp ảo, Discord & OBS (Tab Hướng dẫn)

Tab 2 được tái cấu trúc thành lưới các bước trực quan (Step Grid) 2 cột, loại bỏ văn bản dồn cục, dễ theo dõi:

### 4.1. Sơ đồ luồng tín hiệu (Signal Routing Flow)
```
[Microphone thực] ──➔ [RealTimeVoiceChanger] ──➔ [CABLE Input (Phát ra Cáp ảo)]
                                                         │
                                                         ▼
[Discord / OBS / Game] ◄── [CABLE Output (Thu từ Cáp ảo)] ◄┘
```

### 4.2. Các bước cấu hình chi tiết (4 Bước chuẩn)
- **Bước 1: Cài đặt Driver VB-Audio CABLE**: Tải từ trang chủ VB-Audio, chạy Setup dưới quyền Administrator và khởi động lại Windows.
- **Bước 2: Cấu hình RealTimeVoiceChanger**:
  * Input = Microphone thực (ví dụ: Microphone USB / Realtek Audio).
  * Output = CABLE Input (VB-Audio Virtual Cable).
- **Bước 3: Cấu hình Discord**:
  * Vào *User Settings ➔ Voice & Video*.
  * Input Device = **CABLE Output (VB-Audio Virtual Cable)**.
  * Output Device = Tai nghe thực tế của bạn.
  * **Quan trọng**: Tắt tính năng *Noise Suppression (Krisp)* và *Echo Cancellation* để tránh làm biến dạng âm sắc giọng nữ đã chuyển đổi.
- **Bước 4: Cấu hình OBS Studio**:
  * Thêm nguồn âm thanh *Audio Input Capture*.
  * Chọn Device = **CABLE Output (VB-Audio Virtual Cable)**.
  * Điều chỉnh âm lượng và bật tính năng *Monitor and Output* nếu cần nghe lại.

### 4.3. Bảng xử lý sự cố thường gặp (Troubleshooting FAQ)
- **Sự cố 40 kHz với WASAPI**: Hướng dẫn chuyển sang Host API `[MME]` hoặc `[DirectSound]` nếu Windows từ chối tần số mẫu 40.000 Hz.
- **Tiếng rè hoặc giật cục**: Kiểm tra chỉ số RTF trên Dashboard. Nếu RTF > 1.0, giảm `index_rate` hoặc chuyển sang chế độ hiệu năng cao.
- **Không nghe thấy tiếng**: Kiểm tra đảm bảo không chọn thiết bị Output trùng với Micro hoặc CABLE Output.

---

## 5. Kiến trúc Đa luồng & An toàn Luồng (Threading Architecture)

- **Không chặn GUI Thread**: Mọi tác vụ tốn thời gian như nạp model (`select_profile`), khởi tạo audio stream (`start`), dừng pipeline (`stop`) và giải phóng tài nguyên (`shutdown`) đều được thực thi trên worker thread nền (`threading.Thread`).
- **Đồng bộ hàng đợi an toàn (Queue Marshaling)**: Dữ liệu trạng thái và lỗi được gửi về Tkinter main thread qua `queue.Queue` và xử lý thông qua `root.after()`.
- **Dọn dẹp tài nguyên triệt để**: Bắt sự kiện `WM_DELETE_WINDOW` để gọi `service.shutdown()`, đóng PortAudio streams và giải phóng VRAM GPU trước khi đóng tiến trình.

---

## 6. Kiểm định Chất lượng & Báo cáo Kiểm thử

Hệ thống được kiểm thử tự động toàn diện trong môi trường headless:

```powershell
# Chạy kiểm thử đơn vị GUI
.venv\Scripts\python -m unittest tests/unit/test_gui_app.py tests/unit/test_gui_settings.py

# Chạy toàn bộ test suite dự án
.venv\Scripts\python -m unittest discover -s tests
```

- **Kết quả kiểm thử tự động**: Đạt **100% PASS** trên toàn bộ các bài kiểm thử unit/integration.
- **Kiểm thử co giãn kích thước**: Đã xác nhận giao diện hiển thị ổn định, không phát sinh lỗi layout ở các độ phân giải `960x680`, `1080x760` và `1280x800`.
- **Hạng mục kiểm thử thực địa cần làm với phần cứng thật**:
  1. Kiểm tra micro và tai nghe vật lý đang cắm vào máy tính.
  2. Kiểm tra cuộc gọi đàm thoại thực tế trên Discord thông qua VB-Cable.
