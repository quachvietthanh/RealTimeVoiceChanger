# Hướng dẫn Ứng dụng Desktop GUI — RealTimeVoiceChanger

Tài liệu mô tả cách khởi chạy, bố cục, luồng thao tác và kiểm thử của giao diện desktop của **RealTimeVoiceChanger**. Mọi hành vi mô tả ở đây đã được kiểm tra bằng test tự động không cần GPU/thiết bị âm thanh (xem mục 6); những gì chưa kiểm tra được ghi rõ.

---

## 1. Giới thiệu & Khởi chạy

Desktop GUI dùng `tkinter`/`ttk` của thư viện chuẩn, giao diện tối, DPI-aware trên Windows, và chỉ gọi backend qua `VoiceChangerService` theo `docs/gui-backend-contract.md`. GUI không tự suy luận, không tự đọc registry giọng và không tự quyết định giọng nào nạp được.

```powershell
# Cách 1: launcher
.venv\Scripts\python scripts/run_gui.py

# Cách 2: entry point của package
.venv\Scripts\python -m voice_changer.main --gui

# Nếu đã cài editable (pip install -e .)
voice-changer
```

*(Thêm `--cli` để chạy giao diện dòng lệnh.)*

---

## 2. Thiết kế & hiển thị

### 2.1. Màu, chữ, biểu tượng
- Bảng màu và cỡ chữ nằm trong `src/voice_changer/gui/theme.py` (`Colors`, `Fonts`). Cỡ chữ tính bằng point, không bằng pixel. Font: `Segoe UI`, đường dẫn/trace: `Consolas`.
- **Logo/biểu tượng ứng dụng** là thiết kế gốc của dự án: ô vuông bo góc nền tối với các vạch sóng âm chuyển màu tím → xanh ngọc. Tệp được sinh bằng `scripts/generate_app_icon.py` (Pillow) vào `src/voice_changer/gui/assets/icons/`: `app_icon_{16,20,24,32,40,48,64,128,256}.png`, `app_icon.ico`, `logo.svg`. Cửa sổ dùng các PNG này qua `iconphoto` và `.ico` qua `iconbitmap`; thiếu tệp thì chỉ ghi log và ứng dụng vẫn chạy. Sidebar dùng PNG gần nhất với `px(32)`; nếu không có thì hiển thị chữ "RT".
- **Avatar giọng** là monogram gốc (chữ viết tắt trên nền màu). Không tải ảnh từ mạng, không đóng gói hình ảnh của bên thứ ba. `voice_presentation.json` có thể trỏ tới một PNG cục bộ trong `gui/assets/` (đường dẫn URL, đường dẫn tuyệt đối và đường dẫn ra ngoài thư mục assets đều bị từ chối); hiện chưa có ảnh nào.

### 2.2. DPI & co giãn
- Cửa sổ mặc định `1140 x 780`, tối thiểu `980 x 680` ở 100%.
- `launch_gui()` gọi `SetProcessDpiAwareness(1)` (system-aware) *trước* `tk.Tk()`; mọi kích thước pixel cố định (cửa sổ, sidebar 240, avatar Studio 72, logo, wraplength cố định) đi qua `px()`. Không dùng per-monitor DPI.
- Đã kiểm tra trực quan ở 125% Windows thật **trước** đợt thiết kế lại này; bố cục mới **chưa** được xem trực quan trên màn hình thật (mới kiểm tra bằng test headless và mô phỏng 150% bằng `tk scaling`).
- Trang dài có thanh cuộn; cột Studio căn giữa và không rộng quá `px(760)`.

---

## 3. Điều hướng

Sidebar (chỉ để điều hướng, không hiển thị trạng thái engine):
1. `🎙️ Studio` — chọn giọng, nạp giọng, bắt đầu/dừng.
2. `🎭 Thư viện giọng` — tất cả giọng đã đăng ký và bộ sưu tập dự kiến.
3. `🔌 Âm thanh & Kết nối` — nơi duy nhất chọn micro/ngõ ra, cùng hướng dẫn Discord/OBS/game.
4. (dưới cùng) `🛠️ Nâng cao / Khắc phục sự cố` — số liệu chi tiết, thông số luồng, lỗi gần nhất, file log.

Phần đầu sidebar: logo, tên "RealTimeVoiceChanger" và phiên bản đọc từ metadata gói / `pyproject.toml` (không hardcode). Đầu trang chỉ có tiêu đề và một dòng mô tả của trang hiện tại.

---

## 4. Các trang

### 4.1. Studio (`studio`)
Một cột, theo thứ tự:
1. **Giọng đang chọn** (`ProfilesPanel`): avatar, tên, mô tả ngắn (từ `gui/assets/voice_presentation.json`), nhãn trạng thái, ngôn ngữ, nhãn quyền dữ liệu (nếu registry có), và một câu về khả năng sử dụng. Ô "Giọng:" chỉ liệt kê giọng mà backend báo `selectable` (cộng với giọng đang chọn). Liên kết "Thư viện giọng →".
2. **Thẻ hành động** (`ControlsPanel`): pill trạng thái engine, **một** nút chính đổi theo trạng thái thật:

   | Trạng thái | Giọng đang chọn | Nút |
   |---|---|---|
   | IDLE / READY / ERROR | chưa được nạp, nạp được | **Nạp giọng** |
   | READY | đã nạp (theo `EngineStatus`) | **▶ Bắt đầu chuyển giọng** |
   | RUNNING | bất kỳ | **■ Dừng** (+ nút phụ "Dừng phiên và chuyển sang …" nếu đang chọn giọng khác) |
   | LOADING / STARTING / STOPPING | — | nút bận, bị vô hiệu |
   | ERROR | giọng đã nạp | **↻ Thử lại** (dừng phiên lỗi rồi start lại) |
   | — | đã thu hồi / chưa có mô hình / thiếu tệp | vô hiệu, không bao giờ nạp |

   **Chọn giọng không bao giờ tự nạp.** Chỉ nút "Nạp giọng" (hoặc lần nạp tự động lúc khởi động giọng do backend `resolve_profile_id()` chọn) mới gọi `select_profile()`. Khi đổi giọng trong lúc đang chạy, ứng dụng hỏi xác nhận, kiểm tra tệp giọng mới trước, và chỉ dừng phiên khi kiểm tra đạt.

   Lỗi của thao tác nạp/start/stop và lỗi luồng hiển thị **ngay trong thẻ** (thông báo, gợi ý, "Xem chi tiết kỹ thuật"; lỗi thiết bị có thêm nút "Kiểm tra thiết bị âm thanh"), không bằng hộp thoại. Lỗi được xóa khi người dùng bắt đầu thao tác kế tiếp.

   Dòng hiệu năng ngắn ("ổn định" / "sát giới hạn" / "có gián đoạn") chỉ hiện khi số liệu là **trực tiếp** và đã có ít nhất một chunk; ngưỡng RTF trùng với trang Nâng cao.

   Cuối thẻ là **tóm tắt lộ trình** `🎙 micro → 🔊 ngõ ra` lấy từ thiết bị thực sự đang chọn, kèm giải thích: ngõ ra là cáp ảo → ứng dụng khác chọn micro là đầu thu của cáp (VB-Cable: "CABLE Output"); ngõ ra là loa/tai nghe → chỉ để nghe thử. Thiết bị đã lưu nhưng đang rút được cảnh báo ở đây. Liên kết "Đổi thiết bị →".
3. **Tinh chỉnh giọng** (thu gọn mặc định, tóm tắt "Cao độ … · Index …"): `TuningPanel` gồm Pitch (−24…+24), Index Rate, Filter Radius, Protect, nút "Mặc định". Việc lưu tinh chỉnh vào `gui_settings.json` giữ nguyên như trước (ghi nhưng chưa khôi phục khi mở lại).
4. **Khử tiếng ồn micro** (thu gọn mặc định, tóm tắt "Đang tắt" / "Bật · Vừa (áp dụng khi chạy)" / "Đang lọc · …" / "Đã tự ngắt"): `NoiseSuppressionPanel`, mặc định TẮT theo backend.

### 4.2. Thư viện giọng (`library`)
- Ô tìm kiếm (tên, mô tả ngắn, id, mô tả registry, ngôn ngữ), nút xóa, các chip lọc: Tất cả, Dùng được, Thử nghiệm, Dự kiến, Không khả dụng, Tiếng Việt, Tiếng Anh; dòng đếm "Hiển thị X / Y giọng".
- Thẻ giọng: avatar monogram, tên, ngôn ngữ, mô tả ngắn, câu về khả năng sử dụng, nhãn trạng thái (đã thu hồi / chưa có mô hình / thiếu tệp / thử nghiệm / bản mẫu / đã duyệt / đang dùng — mỗi loại một màu), "Chi tiết" (thông số từ `ProfileSummary`, quyền dữ liệu, ghi chú registry).
- Nút "Dùng trong Studio" chỉ có ở giọng `selectable`; nó **chọn** giọng và chuyển sang Studio, không nạp. Giọng đã thu hồi/chưa có mô hình/thiếu tệp không có nút.
- **Bộ sưu tập VALORANT — dự kiến** (thu gọn, tự mở khi tìm kiếm): 29 nhân vật theo trang chính thức `https://playvalorant.com/en-us/agents/` (kiểm tra ngày 2026-10-23; nhân vật được bên thứ ba báo trước cho 2026-10-27 chưa được thêm). Dữ liệu ở `gui/assets/character_catalog.json`, chỉ để trình bày, avatar là monogram theo màu vai trò, không có hình ảnh của Riot. Nhân vật chưa liên kết hiển thị "Chưa có preset giọng" và không có nút. Liên kết `linked_profile_id` chỉ được chấp nhận khi trùng một hồ sơ đã đăng ký; liên kết sai bị từ chối và ghi log. Hiện chỉ Jett và Sage có liên kết (tới hồ sơ `jett`, `sage` trong registry). Kèm tuyên bố không liên kết với Riot Games.

### 4.3. Âm thanh & Kết nối (`audio_settings`)
- Chọn micro và ngõ ra (nơi duy nhất trong ứng dụng), "Làm mới danh sách", khóa khi đang chạy.
- Lưu thiết bị (`gui_settings.json`, theo tên + host API) **chỉ khi người dùng tự chọn**, và chỉ lưu phía vừa đổi. Tải/làm mới danh sách không bao giờ lưu: thiết bị đã lưu nhưng đang rút được thay tạm bằng thiết bị mặc định cho phiên này kèm cảnh báo; cắm lại rồi "Làm mới" sẽ chọn lại.
- Thẻ lộ trình và lưu ý không chọn loa ngoài làm ngõ ra khi dùng micro (tránh vòng hồi tiếp).
- Các mục hướng dẫn thu gọn: cáp âm thanh ảo (không đóng gói VB-Cable), Discord, OBS, game có voice chat (chọn micro là đầu thu của cáp trong cài đặt âm thanh của game; **không** có tích hợp trực tiếp với game nào), khắc phục sự cố (WASAPI từ chối 40 kHz, giật khi RTF cao, nghe lại).

### 4.4. Nâng cao / Khắc phục sự cố (`diagnostics`)
- **Hiệu năng**: RTF, suy luận trung bình/chunk, độ trễ phần mềm ước tính, buffer phát hiện tại, chunk thành công/rơi, underflow. Nhãn nguồn: "Trực tiếp · phiên #N" hoặc "Phiên trước (#N) · đã kết thúc, không còn cập nhật". Giá trị trung bình trước chunk đầu tiên và buffer sau khi dừng hiển thị "--". Ghi chú: độ trễ phần mềm không gồm độ trễ driver/thiết bị.
- **Thông số luồng**: đọc từ `AudioSettings` của phiên đang chạy (hoặc mặc định cho phiên tiếp theo): độ dài chunk, bộ đệm khởi động, hàng đợi tối đa, ngữ cảnh, crossfade, tìm SOLA; thuật toán F0 từ cấu hình hiện tại; độ trễ ghép chunk + khử ồn từ `algorithmic_delay_ms` của metrics.
- **Lỗi gần nhất** (mã, phân loại, gợi ý, chi tiết kỹ thuật).
- **File log**: đường dẫn, "Sao chép đường dẫn", "Mở thư mục log".

---

## 5. Đa luồng

- Nạp giọng, start, stop và shutdown chạy trên thread nền; kết quả và lỗi được đưa về thread Tk qua `queue.Queue` + `root.after(40, …)`.
- Trạng thái và metrics được thăm dò ~3,3 Hz; GUI chỉ coi một giọng là "đã nạp" khi `EngineStatus.model_loaded` và `profile_id` khớp (không dùng `ProfileSummary.is_active`).
- Đóng cửa sổ gọi `service.shutdown()` trên thread nền rồi mới hủy cửa sổ.

---

## 6. Kiểm thử

Chạy headless, dịch vụ backend được mock (không GPU, không nạp mô hình, không thiết bị âm thanh, không mạng):

```powershell
.venv\Scripts\python -m unittest tests.unit.test_gui_app tests.unit.test_gui_views tests.unit.test_gui_dpi tests.unit.test_gui_settings tests.unit.test_gui_presentation
```

`test_gui_presentation` kiểm tra: bảng trạng thái → nút chính, giọng thu hồi/chưa có mô hình/thiếu tệp không bao giờ nạp được, câu chữ về kiểm tra tệp, danh mục 29 nhân vật và việc từ chối liên kết sai, nhân vật chưa liên kết không có nút, chỉ nhận ảnh cục bộ, logo/avatar dự phòng, lộ trình khớp thiết bị, Studio không có ô chọn thiết bị, số liệu trực tiếp và phiên trước, giới hạn chiều rộng cột Studio, và khởi động ứng dụng khi mọi kết nối mạng bị chặn.

**Chưa kiểm tra** (cần làm trên máy thật):
1. Xem trực quan bố cục mới ở 100/125/150% Windows thật.
2. Micro và tai nghe vật lý; cuộc gọi Discord/OBS/game qua VB-Cable.
3. Biểu tượng trên thanh tác vụ Windows.
