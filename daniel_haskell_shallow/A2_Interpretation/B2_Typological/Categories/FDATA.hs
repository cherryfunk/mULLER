{-# LANGUAGE GADTs #-}

module A2_Interpretation.B2_Typological.Categories.FDATA where

import A2_Interpretation.B2_Typological.Categories.DATA (DATA)

-- | The category FDATA (cartesian closed extension of DATA)
-- FDATA = DATA + exponentials (function objects) + monadic objects.
--
-- DATA has finite products (Unit, Prod).
-- FDATA adds:
--   - Exponentials (function spaces a -> b), making it cartesian closed.
--   - Monadic types (m a), for Kleisli interpretation.
data FDATA a where
  -- | Embed any DATA object into FDATA.
  Embed :: DATA a -> FDATA a
  -- | Exponential object (function space).
  Exp :: FDATA a -> FDATA b -> FDATA (a -> b)
  -- | Monadic object (for Kleisli lifting).
  Monadic :: (Monad m) => FDATA a -> FDATA (m a)
