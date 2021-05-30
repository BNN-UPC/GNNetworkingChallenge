'''
 *
 * Copyright (C) 2021 Universitat Politècnica de Catalunya.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
'''

# -*- coding: utf-8 -*-

import sys
import numpy as np
import ignnition

def normalization(feature, feature_name):
    if feature_name == 'delay':
        feature = np.log(feature)
    return feature

def denormalization(feature, feature_name):
    if feature_name == 'delay':
        feature = np.exp(feature)
    return feature


def main():
    model = ignnition.create_model(model_dir= './')
    model.computational_graph()
    model.train_and_validate()
    predictions = model.predict()
    #Do stuff here
    print(predictions)

if __name__ == "__main__":
        main ()
