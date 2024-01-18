import math

class My_Feature_Encoder:
    def __init__(self, df, feature):
        #Collecting Values
        unique_values = df[feature].unique().tolist()
        if '' not in unique_values:
            unique_values.append('')
        
        #Removing nan as a value from list
        unique_values = [x for x in unique_values if not (isinstance(x, float) and math.isnan(x))]
        
        #Sorting List
        unique_values = sorted(unique_values)

        #Creating a dictionary with values corresponding to the sorted order
        self.value_to_int_dict = {}
        counter = 0
        for value in unique_values:
            self.value_to_int_dict[value] = counter
            counter += 1

    def enc(self, value):
        #if the value is nan (missing), return -1
        if isinstance(value, float):
            if math.isnan(value):
                return -1

        #if the value is not in the dictionary, insert it and place it properly sorted
        if value not in self.value_to_int_dict:
            self.value_to_int_dict[value] = 0
            self.value_to_int_dict = dict(sorted(self.value_to_int_dict.items())) 
            i = list(self.value_to_int_dict.keys()).index(value)
            l_val = list(self.value_to_int_dict.keys())[i-1]
            l_int = self.value_to_int_dict[l_val]
            
            try:
                r_val = list(self.value_to_int_dict.keys())[i+1]
                r_int = self.value_to_int_dict[r_val]
                dif = r_int - l_int
                self.value_to_int_dict[value] = l_int + dif * 0.5
            except Exception as _:
                self.value_to_int_dict[value] = l_int + 1
            
        #Returning the numeric value of the value
        return self.value_to_int_dict[value]  