using Statistics

mutable struct AnalysisWindow
    count::Int64
    sum::Float64
    sumsquares::Float64
    alltimemax::Float64
    alltimemin::Float64
    alltimelatest::Float64
    lastpoptime::DateTime
    maxheap::BinaryMaxHeap{Tuple{Float64,DateTime}}
    minheap::BinaryMinHeap{Tuple{Float64,DateTime}}

    function AnalysisWindow()
        new(0, NaN, NaN, NaN, NaN, NaN, DateTime(Year(0)),
            BinaryMaxHeap{Tuple{Float64,DateTime}}(), BinaryMinHeap{Tuple{Float64,DateTime}}())
    end
end

function Base.push!(window::AnalysisWindow, time::DateTime, datum::Float64)
    # starting from no data requires some extra work, since values start as NaN
    if window.count == 0
        window.count = 1
        window.sum = datum
        window.sumsquares = datum * datum
    else
        window.count += 1
        window.sum += datum
        window.sumsquares += datum * datum
    end

    if !(window.alltimemax ≥ datum) # NB: true if alltime max is NaN
        window.alltimemax = datum
    end

    if !(window.alltimemin ≤ datum) # NB: true if alltime max is NaN
        window.alltimemin = datum
    end

    window.alltimelatest = datum

    DataStructures.push!(window.maxheap, (datum, time))
    DataStructures.push!(window.minheap, (datum, time))
end

function Base.pop!(window::AnalysisWindow, time::DateTime, datum::Float64)
    # if this is the last datapoint reset the sums to NaNs
    @boundscheck(
        if window.count ≤ 0
            throw(BoundsError("Attempt to remove datum from an empty window"))
        end
    )
    if window.count <= 1
        window.count = 0
        window.sum = NaN
        window.sumsquares = NaN
    else
        window.count -= 1
        window.sum -= datum
        window.sumsquares -= datum * datum
    end
    window.lastpoptime = time
end

function Base.empty!(window::AnalysisWindow)
    window.count = 0
    window.sum = NaN
    window.sumsquares = NaN
    window.alltimemin = NaN
    window.alltimemax = NaN
    window.alltimelatest = NaN

    # note: could call empty!(window.maxheap.valtree), but that would bypass the official interface
    while !isempty(window.maxheap)
        DataStructures.pop!(window.maxheap)
    end
    while !isempty(window.minheap)
        DataStructures.pop!(window.minheap)
    end

    window.lastpoptime = DateTime(Year(0))
end

function Statistics.mean(window::AnalysisWindow)
    return window.sum / window.count
end

function Statistics.std(window::AnalysisWindow)
    return window.count < 2 ? NaN : √(
        max(0., window.sumsquares - window.sum*window.sum/window.count) /
            (window.count - 1))
end

function Base.max(window::AnalysisWindow)
    # remove any stale items from the top of the binary heap
    while !isempty(window.maxheap) && window.lastpoptime ≥ first(window.maxheap)[2]
        DataStructures.pop!(window.maxheap)
    end

    if isempty(window.maxheap)
        return NaN
    else
        return first(window.maxheap)[1]
    end
end

function Base.min(window::AnalysisWindow)
    # remove any stale items from the top of the binary heap
    while !isempty(window.minheap) && window.lastpoptime ≥ first(window.minheap)[2]
        DataStructures.pop!(window.minheap)
    end

    if isempty(window.minheap)
        return NaN
    else
        return first(window.minheap)[1]
    end
end

function maxmindiff(window::AnalysisWindow)
    return max(window) - min(window)
end

function alltimemin(window::AnalysisWindow)
    return window.alltimemin
end

function alltimemax(window::AnalysisWindow)
    return window.alltimemax
end

function alltimelatest(window::AnalysisWindow)
    return window.alltimelatest
end

function allstatnames(window::AnalysisWindow)
    return ["mean", "std", "max", "min", "maxmindiff", "alltimemax", "alltimemin", "alltimelatest"]
end

function allstats(window::AnalysisWindow)
    return mean(window), std(window), max(window), min(window), maxmindiff(window),
        alltimemax(window), alltimemin(window), alltimelatest(window)
end

const numanalysiswindowstats = length(allstatnames(AnalysisWindow()))

function allmedstatnames(window::AnalysisWindow)
    return ["totaldose", "maxdose", "alltimemax", "alltimelatest"]
end

function allmedstats(window::AnalysisWindow)
    return [
        window.count == 0 ? 0 : window.sum,
        (x -> x > 0 ? x : 0)(max(window)),
        (x -> x > 0 ? x : 0)(alltimemax(window)),
        (x -> x > 0 ? x : 0)(alltimelatest(window)),
    ]
end

const numanalysiswindowmedstats = length(allmedstatnames(AnalysisWindow()))
