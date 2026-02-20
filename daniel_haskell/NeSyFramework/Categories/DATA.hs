{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module NeSyFramework.Categories.DATA where

import NeSyFramework.Categories.Utils (In, DATA)
import NeSyFramework.Monads.Dist (Dist)
import NeSyFramework.Monads.Giry (Giry)

-- | The category DATA

-- 1. Base objects
instance In DATA Int
instance In DATA Double
instance In DATA Bool
instance In DATA Char
instance In DATA String

-- 2. Finite products
instance In DATA ()
instance (In DATA a, In DATA b) => In DATA (a, b)

-- 3. Finite "sets" (lists over one type, mixed could be added later)
instance In DATA a => In DATA [a]

-- 4. Closure under Monads on DATA
instance In DATA a => In DATA (Dist a)
instance In DATA a => In DATA (Giry a)
