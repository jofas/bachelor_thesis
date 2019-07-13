require 'lfs'

trans = {
  -- data sets
  ["bank"]           = [[\texttt{bank}]],
  ["bankadditional"] = [[\texttt{bank-additional}]],
  ["car"]            = [[\texttt{car}]],
  ["creditcard"]     = [[\texttt{credit card}]],
  ["usps"]           = [[\texttt{usps}]],
  ["winequality"]    = [[\texttt{wine}]],
  -- scoring classifier
  ["cp"] = [[\texttt{cp}]],
  ["rf"] = [[\texttt{rf}]],
  -- reward functions
  ["asymetric_1000"]           = [[\texttt{asymmetric loss 1000}]],
  ["asymetric_200"]            = [[\texttt{asymmetric loss 200}]],
  ["asymetric_50"]             = [[\texttt{asymmetric loss 50}]],
  ["random"]                   = [[\texttt{random loss}]],
  ["random_scaled_1_100"]      = [[\texttt{random loss scaled 1:100}]],
  ["random_scaled_1_20"]       = [[\texttt{random loss scaled 1:20}]],
  ["random_scaled_1_5"]        = [[\texttt{random loss scaled 1:5}]],
  ["simple"]                   = [[\texttt{simple}]],
  ["simple_scaled_1_100"]      = [[\texttt{simple scaled 1:100}]],
  ["simple_scaled_1_20"]       = [[\texttt{simple scaled 1:20}]],
  ["simple_scaled_1_5"]        = [[\texttt{simple scaled 1:5}]],
  ["asymetric_1000_abstain"]   = [[\texttt{asymmetric 1000 abstain}]],
  ["asymetric_200_abstain"]    = [[\texttt{asymmetric 200 abstain}]],
  ["asymetric_50_abstain"]     = [[\texttt{asymmetric 50 abstain}]],
  ["random_gain"]              = [[\texttt{random}]],
  ["random_gain_scaled_1_100"] = [[\texttt{random scaled 1:100}]],
  ["random_gain_scaled_1_20"]  = [[\texttt{random scaled 1:20}]],
  ["random_gain_scaled_1_5"]   = [[\texttt{random scaled 1:5}]],
  -- regressors
  ["GP [1,2]"]     = [[\texttt{GP [1, 2]}]],
  ["GP [1e-1,1]"]  = [[\texttt{GP [1e-1, 1]}]],
  ["GP [1e-3, 1]"] = [[\texttt{GP [1e-3, 1]}]],
  ["SVR RBF C100"] = [[\texttt{SVR C100}]],
  ["SVR RBF C1"]   = [[\texttt{SVR C1}]],
  ["bare"]         = [[\texttt{bare}]]
}

idx_reg = {
  ["GP [1,2]"]     = 4,
  ["GP [1e-1,1]"]  = 5,
  ["GP [1e-3, 1]"] = 6,
  ["SVR RBF C100"] = 3,
  ["SVR RBF C1"]   = 2,
  ["bare"]         = 1
}

idx_reward = {
  ["asymetric_1000"]           = 7,
  ["asymetric_200"]            = 6,
  ["asymetric_50"]             = 5,
  ["random"]                   = 8,
  ["random_scaled_1_100"]      = 11,
  ["random_scaled_1_20"]       = 10,
  ["random_scaled_1_5"]        = 9,
  ["simple"]                   = 1,
  ["simple_scaled_1_100"]      = 4,
  ["simple_scaled_1_20"]       = 3,
  ["simple_scaled_1_5"]        = 2,
  ["asymetric_1000_abstain"]   = 18,
  ["asymetric_200_abstain"]    = 17,
  ["asymetric_50_abstain"]     = 16,
  ["random_gain"]              = 12,
  ["random_gain_scaled_1_100"] = 15,
  ["random_gain_scaled_1_20"]  = 14,
  ["random_gain_scaled_1_5"]   = 13,
}

local root = [[../experiments/]]

local function iter_dir(path, callback, kwargs)
  for entry in lfs.dir(path) do
    if entry ~= "." and entry ~= ".." then
      callback(entry, kwargs)
    end
  end
end

local function reward_fn(reward_fn, kwargs)
  if
  kwargs.result
    [kwargs.curr_ds]
    [kwargs.curr_scorer]
    [reward_fn] == nil
  then
    kwargs.result
      [kwargs.curr_ds]
      [kwargs.curr_scorer]
      [reward_fn] = {}
  end

  local i = 0
  for line in io.lines(
    kwargs.dir.."/"..reward_fn.."/result.csv"
  ) do
    if i ~= 0 then
      local reg, reward, _ = string.match(
        line, "([%w%[%]-, ]*);(%d.[%de-]*);(%d.%d*)"
      )

      kwargs.result
        [kwargs.curr_ds]
        [kwargs.curr_scorer]
        [reward_fn]
        [reg] = tonumber(reward)
    end
    i = i + 1
  end
end

local function scorer(scorer, kwargs)
  if kwargs.result[kwargs.curr_ds][scorer] == nil then
    kwargs.result[kwargs.curr_ds][scorer] = {}
  end
  kwargs.curr_scorer = scorer
  kwargs.dir = kwargs.dir.."/"..scorer

  iter_dir(kwargs.dir, reward_fn, kwargs)

  kwargs.dir = string.gsub(kwargs.dir, "/"..scorer, "")
end

local function data_set(name, kwargs)
  ds = string.gsub(name, "_%d*", "")
  if kwargs.result[ds] == nil then
    kwargs.result[ds] = {}
  end
  kwargs.curr_ds = ds
  kwargs.dir = root..name

  iter_dir(kwargs.dir, scorer, kwargs)
end

local function len(table)
  local count = 0
  for _ in pairs(table) do count = count + 1 end
  return count
end

local function result_tables()
  local kwargs = {result = {}}
  iter_dir(root, data_set, kwargs)

  local res = kwargs.result
  for ds, v0 in pairs(res) do
    for scorer, v1 in pairs(v0) do

      local r = {{}}
      local i = 0

      for reward, v2 in pairs(v1) do
        local line = {trans[reward]}

        for reg, rew in pairs(v2) do
          if i == 0 then
            --r[1][idx_reg[reg]] = "&"..trans[reg].." "
            r[1][idx_reg[reg]] = ";"..reg
          end

          line[idx_reg[reg] + 1] =
            ";"..string.format("%.3f", rew)
            --" &"..string.format("%.3f", rew)
        end

        i = i + 1
        r[idx_reward[reward] + 1] = line
      end

      print(ds.." "..scorer)
      for _, x in pairs(r) do
        local line = ""
        for _, y in pairs(x) do
          line = line..y
        end
        print(line)
      end

      --print("\n"..ds.." "..scorer.."\n")

      --print([[\begin{table}]])
      --print([[{\scriptsize]])
      --print([[\begin{tabu}{l|l|l|l|l|l|l}]])
      --for _, x in pairs(r) do
      --  local line = ""
      --  for _, y in pairs(x) do
      --    line = line..y
      --  end
      --  print(line..[[ \\]])
      --end
      --print([[\end{tabu} }]])
      --print([[\caption{ }]])
      --print([[\end{table}]])
    end
  end
end

result_tables()

return { result_tables = result_tables }
