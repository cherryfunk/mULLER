import sys
from pymonad.monad import Monad
from .aggr2sgrpblat import Aggr2SGrpBLat
from .boolean import (
    NonDeterministicBooleanLogic,
    ProbabilisticBooleanLogic,
    ClassicalBooleanLogic
)
from .priest import (
    NonDeterministicPriestLogic,
    ProbabilisticPriestLogic,
    ClassicalPriestLogic
)


def get_logic[T: Monad, O](monad_type: type[T], omega: type[O]) -> Aggr2SGrpBLat[Monad[O]]:
    """
    Get the logic for a specific monad and type.
    """
    current_module = sys.modules[__name__]
    
    # Search through all attributes in the current module
    for attr_name in dir(current_module):
        attr = getattr(current_module, attr_name)
        
        # Check if it's a class and not a built-in or imported type
        if isinstance(attr, type) and hasattr(attr, '__module__'):
            # Check if it's defined in one of our logic modules
            if (attr.__module__.startswith('muller.logics.') and 
                attr.__module__ != 'muller.logics.aggr2sgrpblat'):
                
                # Check if it's a subclass of Aggr2SGrpBLat
                if (hasattr(attr, '__orig_bases__') and 
                    len(attr.__orig_bases__) > 0):
                    
                    base = attr.__orig_bases__[0]
                    
                    # Check if the base is Aggr2SGrpBLat with a generic parameter
                    if (hasattr(base, '__origin__') and 
                        base.__origin__ is Aggr2SGrpBLat and
                        hasattr(base, '__args__') and 
                        len(base.__args__) > 0):
                        
                        monad_arg = base.__args__[0]
                        
                        # Check if the monad type matches our target T
                        if (hasattr(monad_arg, '__origin__') and 
                            monad_arg.__origin__ is monad_type and
                            hasattr(monad_arg, '__args__') and 
                            len(monad_arg.__args__) > 0 and
                            monad_arg.__args__[0] is omega):
                            
                            # Return an instance of the matching class
                            return attr()
    
    # If no matching class is found, raise an exception
    raise ValueError(f"No logic class found for monad type {monad_type} with omega type {omega}")
    

__all__= [
    "Aggr2SGrpBLat",
    "get_logic",
    "NonDeterministicBooleanLogic",
    "ProbabilisticBooleanLogic",
    "ClassicalBooleanLogic",
    "NonDeterministicPriestLogic",
    "ProbabilisticPriestLogic",
    "ClassicalPriestLogic"
]