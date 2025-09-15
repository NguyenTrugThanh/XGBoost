## Bài toán là gì?
Trong lĩnh vực **tài chính – ngân hàng**, một bài toán quan trọng là **Credit Scoring**:  
> Dự đoán khả năng một khách hàng **có vỡ nợ (default)** trong vòng 2 năm tới hay không, dựa trên các thông tin tài chính và lịch sử tín dụng.

- Nếu mô hình dự đoán khách hàng có nguy cơ cao → ngân hàng có thể từ chối hoặc điều chỉnh hạn mức vay.  
- Nếu mô hình dự đoán khách hàng an toàn → có thể chấp nhận cho vay với điều kiện ưu đãi.  

Dữ liệu được sử dụng là bộ **[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)**, gồm ~150.000 khách hàng ở Mỹ với 10 đặc trưng chính:
1. `RevolvingUtilizationOfUnsecuredLines` – tỷ lệ sử dụng tín dụng không thế chấp  
2. `age` – tuổi  
3. `NumberOfTime30-59DaysPastDueNotWorse` – số lần trả chậm 30–59 ngày  
4. `DebtRatio` – tỷ lệ nợ / thu nhập  
5. `MonthlyIncome` – thu nhập hàng tháng  
6. `NumberOfOpenCreditLinesAndLoans` – số khoản vay/tín dụng đang mở  
7. `NumberOfTimes90DaysLate` – số lần trả chậm ≥90 ngày  
8. `NumberRealEstateLoansOrLines` – số khoản vay bất động sản  
9. `NumberOfTime60-89DaysPastDueNotWorse` – số lần trả chậm 60–89 ngày  
10. `NumberOfDependents` – số người phụ thuộc  

Nhãn dự đoán (`TARGET`):  
- **0 = An toàn**  
- **1 = Vỡ nợ trong 2 năm tới**

---

## Thuật toán sử dụng
Dự án sử dụng **XGBoost (Extreme Gradient Boosting)** – một thuật toán ensemble mạnh mẽ cho bài toán phân loại nhị phân, đặc biệt hiệu quả trên dữ liệu tabular.  

Các bước chính:  
1. **Tiền xử lý dữ liệu**:
   - Xử lý giá trị khuyết (`SimpleImputer`).  
   - Chuẩn hóa (`StandardScaler`).  
   - Tạo thêm feature mới: `Income_per_person`, `Debt_to_income_ratio`, và cờ `missing`.  

2. **Huấn luyện mô hình**:
   - Sử dụng `XGBClassifier` với hàm mục tiêu `binary:logistic`.  
   - Tối ưu siêu tham số bằng **Optuna** (bayesian optimization).  
   - Sử dụng `StratifiedKFold` để giữ cân bằng dữ liệu khi cross-validation.  

3. **Hiệu chỉnh xác suất (Calibration)**:
   - Dùng `CalibratedClassifierCV` để xác suất dự đoán chính xác hơn.  

4. **Đánh giá mô hình**:
   - ROC-AUC  
   - Confusion Matrix  
   - Classification Report (Precision, Recall, F1-score)  

---

## Kết quả đạt được
<img width="515" height="1026" alt="image" src="https://github.com/user-attachments/assets/372e538e-0685-49fe-98ec-50e32fe4b61e" />
<img width="472" height="1004" alt="image" src="https://github.com/user-attachments/assets/5d02d342-3f6b-454b-8b3e-22735b186d67" />
<img width="463" height="995" alt="image" src="https://github.com/user-attachments/assets/98fe91c8-3521-4cf5-9bbe-8b27e82bc91c" />
