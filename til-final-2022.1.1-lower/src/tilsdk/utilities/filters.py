from typing import TypeVar, Generic, Sequence
from collections import deque

_T = TypeVar('_T') # Generic type


class SimpleMovingAverage(Generic[_T]):
    '''Simple Moving Average filter.'''

    def __init__(self, n:int, elements:Sequence[_T]=[]):
        '''   
        Parameters
        ----------
        n : int
            Size of averaging window.
        elements : Sequence[_T]
            Initial sequence of elements.
        '''
        self._deque = deque(elements, maxlen=n)
        self._value = None

    def update(self, p:_T) -> _T:
        '''
        Update filter with new reading.
        
        Parameters
        ----------
        p : _T
            New value.

        Returns
        -------
        value : _T
            Filtered value.
        '''
        n = len(self._deque)

        if n < self._deque.maxlen:
            # not full
            self._value = (self._value*n + p)/(n+1) if self._value is not None else p
        else:
            self._value += (p - self._deque.popleft())/self._deque.maxlen

        self._deque.append(p)
        
        return self._value

    def get_value(self) -> _T:
        '''
        Get filtered value.

        Returns
        -------
        value: _T
            Filtered value.
        '''
        return self._value

    def clear(self):
        '''
        Clear filter.
        '''
        self._deque.clear()

    def __len__(self):
        return len(self._deque)

    def is_full(self) -> bool:
        '''
        Check if filter is fully populated.

        Returns
        -------
        is_full : bool
            True if full, False otherwise.
        '''
        return len(self._deque) == self._deque.maxlen