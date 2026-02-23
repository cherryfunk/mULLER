module NeSySystem.Signatures.Dice (diceSig) where

import Syntax (Signature (..))

-- | The Dice signature
-- Sorts: DieResult
-- Rels: ==, even
-- MFuncs: die
diceSig :: Signature
diceSig =
  Signature
    { sortDecls = ["DieResult"],
      funDecls =
        [ ("val_1", [], "DieResult"),
          ("val_2", [], "DieResult"),
          ("val_3", [], "DieResult"),
          ("val_4", [], "DieResult"),
          ("val_5", [], "DieResult"),
          ("val_6", [], "DieResult")
        ],
      relDecls =
        [ ("==", ["DieResult", "DieResult"]),
          ("even", ["DieResult"])
        ],
      mFunDecls =
        [ ("die", [], "DieResult")
        ],
      mRelDecls = []
    }
