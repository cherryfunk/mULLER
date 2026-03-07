{-# LANGUAGE AllowAmbiguousTypes #-}

module A1_Syntax.B4_NonLogical.Dice_Vocab where

-- | Non-Logical Vocabulary for the Dice domain.

-- | Sorts:
type DieResult = Int

-- | Signature:
class Dice_Vocab m where
  -- mFun (Kleisli):
  die :: m DieResult
