

class node:
    def __init__(self, data):
        self._data = data
        self._child = []
        self._label = -1

    def getdata(self):
        return self._data
        
    def getchild(self):
        return self._child
    
    def getlabel(self):
        return self._label
    
    def add_child(self, node):
        self._child.append(node)
        
    def find_data(self, data):
        for child in self._child:
            if child.getdata() == data:
                return child
        return None
        
        
