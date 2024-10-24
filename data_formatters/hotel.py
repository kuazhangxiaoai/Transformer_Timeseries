# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for Electricity dataset.

Defines dataset specific column definitions and data transformations. Uses
entity specific z-score normalization.
"""

import data_formatters.base
import data_formatters.utils as utils
import pandas as pd
import sklearn.preprocessing
from data_formatters.base import set_occupytions_of_date
from datetime import datetime, timedelta
from chinese_calendar import is_holiday
GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

class HotelFormatter(GenericDataFormatter):
    """Defines and formats data for the electricity dataset.

      Note that per-entity z-score normalization is used here, and is implemented
      across functions.

      Attributes:
        column_definition: Defines input and data type of column used in the
          experiment.
        identifiers: Entity identifiers used in experiments.
      """

    _column_definition = [
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('date', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('season', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('step_num', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hotel_occupytion', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('is_holiday', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
    ]
    def __init__(self, start):
        self.start = start
        self.id = 'HU_001'

    def category_filter(self, df:pd.DataFrame, categories):
        filtered = df[df['room_category_label'].isin(categories)]
        return filtered

    def share_amount_filter(self, df):
        filtered = df[df['share_amount'].astype(float) > 0.0]
        return filtered

    def status_filter(self, df, status):
        filtered = df[df['resv_status'].isin(status)]
        return filtered

    def format_predictions(self, df):
        return

    def get_fixed_params(self):
        return

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

            Args:
              df: Data to use to calibrate scalers.
            """
        print('Setting scalers with training data...')
        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(
            InputTypes.ID,
            column_definitions
        )
        target_column = utils.get_single_col_by_input_type(
            InputTypes.TARGET,
            column_definitions
        )
        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME}
        )

        self._real_scalers = {}
        self._target_scaler = {}


        return

    def split_sub_data(self, df, begin, end):
        dates, occupytions, is_holidays, day_of_weeks = [], [], [], []
        step_num_list, months, seasons = [],[],[]
        cursor_date = begin

        while (cursor_date < end):
            dates.append(cursor_date)
            step_num_list.append((cursor_date - self.start).days)
            months.append(cursor_date.month)
            seasons.append((cursor_date.month - 1) // 3 + 1)
            is_holidays.append(is_holiday(cursor_date))
            day_of_weeks.append(cursor_date.isoweekday())
            cursor_date = cursor_date + timedelta(days=1)

        occupytions = set_occupytions_of_date(df, dates)

        return dates, step_num_list, months, seasons, occupytions, is_holidays, day_of_weeks

    def split_data(self, df, train_begin, train_end, val_begin, val_end, test_begin, test_end):
        train_dates,train_steps, train_months, train_seasons, train_occupytions, train_is_holidays, train_day_of_weeks = self.split_sub_data(
            df, train_begin, train_end
        )

        val_dates, val_steps, val_months, val_seasons,val_occupytions, val_is_holidays, val_day_of_weeks = self.split_sub_data(
            df, val_begin, val_end
        )

        test_dates,test_steps, test_months, test_seasons, test_occupytions, test_is_holidays, test_day_of_weeks = self.split_sub_data(
            df, test_begin, test_end
        )

        trainset = {
            'id':self.id,
            'dates': train_dates,
            'step': train_steps,
            'month': train_months,
            'season': train_seasons,
            'occupytion': train_occupytions,
            'is_holiday': train_is_holidays,
            'day_of_week': train_day_of_weeks
        }

        valset = {
            'id': self.id,
            'dates': val_dates,
            'step': val_steps,
            'month': val_months,
            'season': val_seasons,
            'occupytion': val_occupytions,
            'is_holiday': val_is_holidays,
            'day_of_week': val_day_of_weeks
        }

        testset = {
            'id': self.id,
            'dates': test_dates,
            'step': test_steps,
            'month': test_months,
            'season': test_seasons,
            'occupytion': test_occupytions,
            'is_holiday': test_is_holidays,
            'day_of_week': test_day_of_weeks
        }

        trainset = pd.DataFrame(trainset)
        valset = pd.DataFrame(valset)
        testset = pd.DataFrame(testset)

        return trainset, valset, testset

    def concat(self, trainset, valset, testset):
        return pd.concat([trainset, valset, testset], axis=0)

    def save(self, df:pd.DataFrame, savepath):
        df.to_csv(savepath, encoding='utf-8', index=False)

    def transform_inputs(self):
        return
