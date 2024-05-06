# ML_Churn-Analysis-of-E-Commerce
Utilizing supervised learning models to predict customer churn, and discovering customer's behaviour in order to identify the key indicators of customer's churn status

## I. Introduction
### 1. Context
Customer Churn is when customers stop purchasing/using business's products or service in certain period of time. Customer churn is one critical metric because it's less expensive to retain existing customers than acquire new customers. In this kernel, I'll analyze E-commerce customer churn rate and looking for user patterns whos likely churned. Customer churn rate indicates how many existing customers are not using products-services or switch to business competitors.

### 2. Business question
One ecommerce company has a project on predicting churned users in order to offer potential promotions. Specifically, there are 2 main questions that I need to answer:
- What are the patterns/behavior of churned users? What are your suggestions to the company to reduce churned users.
- Build the Machine Learning model for predicting churned users.

### 3. Dataset
<img width="534" alt="Screenshot_10" src="https://github.com/NuongLe123/Python_RFM_analysis/assets/168357450/6c0cd1eb-6c5c-4bda-996c-27a85ccc7e4e">

### 4. Method
Supervised learning with Scikit-learn on Python

- Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately.
- As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process.
- Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.

## II. Proccess
### STEP 1: DATA PROCESSING:

- **OUTLIERS**: Visualization to detect outliers - Handle outliers using IQR method
- **MISSING VALUES**: Handling it by replacing with median:
- **EDA**: Seaborn and Matplotlib were used to conduct Exploratory Data Analysis (EDA) and extract knowledge from the data.
   
### STEP 2: FEATURE TRANSFORMING:
- The dummy variables have been created for the remaining categorical columns so that their input should be considered by the Machine Learning model.

### STEP 3: MODEL SELECTION:

Decide to choose 3 models to apply: Logistic Regression, Decision Tree, Random Forest:
- **Linear model**: Linear regression và Logistic Regression đều là linear model. Tức là với 2 model này, mình cần phải **đưa những features mà nó thật sự có mối quan hệ với cột target column** = cái cột mà mình còn dự đoán. Nó phải thực sự có tương quan thì khi đó cái model này mới hoạt động tốt, nó mới cho ra được 1 cái accuracy cao. Nhược điểm là **bước EDA phải làm cẩn thận**.
- **Non-linear model**: Tuy nhiên, đối với 2 model là decision tree & random forest, thì lại là non-linear model, tức là **có thể đưa toàn bộ features vào**, thì 2 model này nó sẽ tự động lọc ra được những cái features nào mà có tương quan cao nhất, thì nó sẽ lấy cái feature đó vào model. 2 cái model này nó học dựa trên cái tree-based. 
- **Random forest**: Mô hình được khuyến khích dùng là Random forest, do nó **giải quyết được vấn đề Overfitting**, vốn là nhược điểm của Decision tree.

### STEP 4: MODEL TRAINING:
- Split data into train/test set – Fit the model on train set & predict on test set:

### STEP 5: MODEL EVALUATION:

**Nhận xét về 3 models:**
- **model 'Decision Tree'**: có chỉ số Balanced accuracy giữa train test và test set bị chênh lệch quá mức, thể hiện nó bị overfitting (1 và 0.95)
  => ko nên dùng model này.
- **model 'Random Forest'**: có chỉ số Balanced accuracy giữa train test và test set tương tự nhau (0.66 và 0.65), thể hiện model nắm bắt được underlying patterns trong bộ dataset, và đưa ra những dự đoán hợp lý. -> nên dùng model này.
- **model 'Logistic Regression'**: model đang phân loại chính xác trung bình khoảng 60 và 64% điểm dữ liệu trong tập train và tập test (0.60 và 0.64)
- **Kết luận: nên dùng model 'Random Forest'** vì nó có chỉ số Balanced accuracy giữa train test và test set cao và tương tự nhau.

### Recommendations:
- Từ hình vẽ trên, ta có thể thấy **Top 4 predictors of churn là: Tenure, Complain, CashbackAmount, DaySinceLastOrderr**. Do đó, ta sẽ có những recommendations sau cho doanh nghiệp:
- Tăng tenure bằng cách: bắt đầu 1 số **chương trình cho khách hàng thân thiết**, như giảm giá cho họ.
- Complains đứng ở vị trí thứ 2, nên doanh nghiệp cần phải đảm bảo **dịch vụ khách hàng** của họ có đủ trình độ để giải quyết các khiếu nại 1 cách chuyên nghiệp.
- Cung cấp **khuyến mãi cho chủ thẻ tín dụng và thẻ ghi nợ**, vì khách hàng sử dụng thẻ có nhiều khả năng rời bỏ hơn các khách hàng còn lại.
- Ngoài ra, để tăng sự hài lòng của khác hàng,  công ty **cần thực hiện 1 số khảo sát để lấy phản hồi của khách hàng**, từ đó biết được vấn đề cần tập trung và giải quyết nó.
- **Tiến hành thử nghiệmm A/B** để nâng cao trải nghiệm người dùng và giao diện người dùng, điều này sẽ nâng cao trải nghiệm khách hàng và tăng tỷ lệ chuyển đổi.

### STEP 6: IMPROVING MODEL ACCURACY BY HYPERPARAMETER TUNING:
- Nhờ có pp HYPERPARAMETER TUNING, mà accuracy của model 'Random Forest' đã tăng từ 0.66 lên thành 0.96.
