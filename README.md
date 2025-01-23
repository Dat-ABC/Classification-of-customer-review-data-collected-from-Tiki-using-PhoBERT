# Classification of customer review data
## Overview

This project focuses on classifying customer review data collected from Tiki, an e-commerce platform. The reviews are written by customers after purchasing products, making the dataset highly unstructured and imbalanced. Key challenges include:

- **Data Imbalance**: Certain classes dominate the dataset, requiring strategies such as adjusting class weights in the loss function.
- **Complex Text Characteristics**: The reviews contain abbreviations, slang, foreign languages, advertisements, icons, and links, necessitating extensive preprocessing efforts.
- **Data Augmentation**: Various techniques were applied to expand the dataset and address imbalance, improving model robustness.

The project involved training and evaluating multiple Machine Learning (ML) and Deep Learning (DL) models. The **PhoBERT model**, fine-tuned specifically for the Vietnamese language, achieved the best performance in classifying customer reviews accurately.

This project highlights the importance of thorough preprocessing, effective data augmentation, and selecting the right model to handle real-world, noisy, and imbalanced textual data.


## Illustration

| Before data augmentation | After data augmentation |
|--------------------------|-------------------------|
| ![illustration](Images/The%20number%20of%20sentences%20for%20each%20label.png) | ![illustration](Images/Ảnh%20màn%20hình%202025-01-23%20lúc%2015.12.10.png) |

| Frequency of each word | Frequency length of sentence |
|------------------------|-----------------------------|
| ![illustration](Images/Frequency%20of%20each%20word.png) | ![illustration](Images/Ảnh%20màn%20hình%202025-01-23%20lúc%2015.10.19.png) |

| Results of models without data augmentation | Results of models with data augmentation |
|---------------------------------------------|-------------------------------------------|
| ![illustration](Images/Ảnh%20màn%20hình%202025-01-23%20lúc%2015.06.04.png) | ![illustration](Images/Ảnh%20màn%20hình%202025-01-23%20lúc%2015.06.30.png) |

| Results of models without data augmentation |
|---------------------------------------------|
| ![illustration](Images/Ảnh%20màn%20hình%202025-01-23%20lúc%2016.28.21.png) |

## Contact
For any inquiries or support, please contact nguyendatkak@gmail.com.
