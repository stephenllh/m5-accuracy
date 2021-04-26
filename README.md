# Silver Medal Solution of M5 Forecasting - Accuracy Kaggle Competition

<!-- ABOUT THE PROJECT -->
## About The Project

<br/>
Given hierarchical sales data from Walmart, the world’s largest company by revenue, we need to forecast daily sales for the next 28 days.
<br/>

<p align="center">
  <img src="/image/image.png" alt="Competition image" width="800" height="400"/>
</p>


This is my solution to the [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) Kaggle competition. I used a LightGBM to train on the tabular dataset, which was preprocessed to include 7-day and 28-day rolling mean features.

Result: Weighted root mean squared scaled error (RMSSE) score of 0.63730
 in the [private leaderboard](https://www.kaggle.com/c/m5-forecasting-accuracy/leaderboard). Ranked 248 out of 5558 teams (Top 5% - silver medal).
<br/><br/>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple example steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/stephenllh/m5_accuracy.git
   ```

1. Change directory
   ```sh
   cd m5_accuracy
   ```

2. Install packages
   ```sh
   pip install requirements.txt
   ```

<br/>

<!-- USAGE EXAMPLES -->
## Usage

1. Change directory
   ```sh
   cd m5_accuracy
   ```

2. Create a directory called `input`
   ```sh
   mkdir input
   cd input
   ```

3. Download the dataset into the folder
    - Option 1: Use Kaggle API
      - `pip install kaggle`
      - `kaggle competitions download -c m5-forecasting-accuracy`
    - Option 2: Download the dataset from the [competition website](https://www.kaggle.com/c/m5-forecasting-accuracy/data).

4. Run the training script
   ```sh
   cd ..
   python train.py
   ```

5. (Optional) Run the inference script
   ```sh
   python inference.py
   ```

<br/>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
<br></br>


<!-- CONTACT -->
## Contact

Stephen Lau - [Email](stephenlaulh@gmail.com) - [Twitter](https://twitter.com/StephenLLH) - [Kaggle](https://www.kaggle.com/faraksuli)


