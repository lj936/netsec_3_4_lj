import math

class My_Feature_Encoder:
    def __init__(self, xdf, feature):
        unique_values = xdf[feature].unique().tolist()
        if '' not in unique_values:
            unique_values.append('')
        
        unique_values = [x for x in unique_values if not (isinstance(x, float) and math.isnan(x))]

        unique_values = sorted(unique_values)

        self.value_to_int_dict = {}
        counter = 0
        for value in unique_values:
            self.value_to_int_dict[value] = counter
            counter += 1
        
        #self.value_to_int_dict = {float("nan"): -1, **self.value_to_int_dict}
        #print(self.value_to_int_dict)

    def enc(self, value):
        if isinstance(value, float):
            if math.isnan(value):
                return -1

        if value not in self.value_to_int_dict:
    
            self.value_to_int_dict[value] = 0
            
            self.value_to_int_dict = dict(sorted(self.value_to_int_dict.items())) 
            
            i = list(self.value_to_int_dict.keys()).index(value) #Index von value finden
            l_val = list(self.value_to_int_dict.keys())[i-1] #Linker Nachbar - Value
            l_int = self.value_to_int_dict[l_val] #Linker Nachbar - Integer
            
            try:
                r_val = list(self.value_to_int_dict.keys())[i+1] #Rechter Nachbar - Value
                r_int = self.value_to_int_dict[r_val] #Rechter Nachbar - Integer

                dif = r_int - l_int

                self.value_to_int_dict[value] = l_int + dif * 0.5
            
            except Exception as _:
                self.value_to_int_dict[value] = l_int + 1

                #print(value, ":", self.value_to_int_dict[value]) #TODO: DELETE
            
            

        return self.value_to_int_dict[value]  