use crate::{load_hlcv, plot, Account, RsiState};
use cmaes::{CMAESOptions, ParallelObjectiveFunction, PlotOptions};
use cmaes::{DVector, Mode, ObjectiveFunction};
use merovingian::hlcv::{Hlcv, MappedHlcvs};
use mouse::prelude::*;
use mouse::sync::{Mutex, RwLock, RwLockReadGuard};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use simplers_optimization::Optimizer;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const MIN_TIMEFRAME: u32 = 60;
const TIMEFRAME_RANGE: u32 = 60;
const TIMEFRAME_STEP: u32 = 60;
const MAX_TIMEFRAME: u32 = MIN_TIMEFRAME + TIMEFRAME_RANGE * TIMEFRAME_STEP;
const MAX_LEN: u32 = 100;
const MAX_HLINE: u32 = 101;
const MAX_LLINE: u32 = 101;

pub struct Context {
    pub timeframe_cache: RwLock<HashMap<u32, Arc<Vec<Hlcv>>>>,
    pub timeframe: AtomicU32,
    pub hybrid: bool,
    pub compute_cost: AtomicU64,
    pub cache_cost: AtomicU64,
    pub timeframe_cost: AtomicU64,
    pub cache_timeframe: bool,
    pub hlcvs_start_ts: u32,
    pub max_risk: f32,
    pub micros_compute: AtomicU64,
    pub micros_cache: AtomicU64,
    pub micros_timeframe: AtomicU64,
    pub n_shards: usize,
    pub shard_i: usize,
    // top 10 najboljih parametra
    pub top: Mutex<Vec<Pair>>,
}

impl Context {
    pub async fn new() -> Result<Self> {
        let start_ts = 1443184345;
        let count = 1633869265 - 1443184345;
        //        let count = 100000;
        //        let count = count / 1000;
        let hlcvs_mapped = load_hlcv("BitMEX", "XBTUSD", start_ts, count).await?;
        let mut cache = HashMap::new();
        let mut hlcvs = Vec::new();
        hlcvs.extend_from_slice(hlcvs_mapped.as_ref());
        cache.insert(1, Arc::new(hlcvs));
        Ok(Self {
            hlcvs_start_ts: hlcvs_mapped.start_ts,
            max_risk: 0.01,
            timeframe: Default::default(),
            hybrid: false,
            micros_compute: Default::default(),
            micros_cache: Default::default(),
            micros_timeframe: Default::default(),
            n_shards: 0,
            timeframe_cache: RwLock::new(cache),
            compute_cost: Default::default(),
            cache_cost: Default::default(),
            timeframe_cost: Default::default(),
            cache_timeframe: false,
            top: Mutex::new(vec![Default::default(); 10]),
            shard_i: 0,
        })
    }

    fn get_hlcvs<'a>(&'a self, timeframe: u32) -> Arc<Vec<Hlcv>> {
        if timeframe == 1 {
            return self.timeframe_cache.read()[&1].clone();
        }
        if self.cache_timeframe {
            let guard = self.timeframe_cache.read();
            if guard.contains_key(&timeframe) {
                return guard[&timeframe].clone();
            }
            drop(guard);
            let hlcvs = self.build_hlcvs(timeframe);
            let mut guard = self.timeframe_cache.write();
            guard.insert(timeframe, Arc::new(hlcvs));
            return guard[&timeframe].clone();
        }
        let hlcvs = self.build_hlcvs(timeframe);
        let mut guard = self.timeframe_cache.write();
        guard.insert(0, Arc::new(hlcvs));
        return guard[&0].clone();
    }

    pub fn build_hlcvs(&self, timeframe: u32) -> Vec<Hlcv> {
        let mut changed_hlcvs = vec![];
        let guard = self.timeframe_cache.read();
        let src = &guard[&1][..];
        let now = Instant::now();
        changed_hlcvs = vec![
            Default::default();
            merovingian::hlcv::change_timeframe_dest_len(
                src.len(),
                self.hlcvs_start_ts,
                1,
                timeframe,
            )
        ];
        merovingian::hlcv::change_timeframe(
            &src,
            self.hlcvs_start_ts,
            1,
            timeframe,
            &mut changed_hlcvs,
        );
        self.timeframe_cost
            .fetch_add(src.len() as u64, Ordering::SeqCst);
        self.micros_timeframe
            .fetch_add(now.elapsed().as_micros() as u64, Ordering::SeqCst);
        changed_hlcvs
    }

    pub fn backtest(&self, timeframe: u32, period: usize, hline: f32, lline: f32) -> Account {
        //        println!("{}", timeframe);
        let mut nanos_cache = 0u128;
        let mut nanos_compute = 0u128;
        let hlcvs = self.get_hlcvs(timeframe);
        let hlcvs = &hlcvs[..];
        let mut now = Instant::now();
        let mut state = RsiState::new(period, &hlcvs);
        let mut offset = state.update_at();
        let mut prev_rsi = state.update(hlcvs, offset);
        nanos_cache += now.elapsed().as_nanos();
        let mut prev_close = hlcvs[offset].close;
        offset += 1;
        let mut account = Account::new();
        for i in offset..hlcvs.len() {
            let mut now = Instant::now();
            let rsi = state.update(hlcvs, i);
            nanos_cache += now.elapsed().as_nanos();
            let mut now = Instant::now();
            account.update(
                &hlcvs,
                i,
                prev_close,
                prev_rsi,
                rsi,
                self.max_risk,
                hline,
                lline,
            );
            prev_close = hlcvs[i].close;
            prev_rsi = rsi;
            nanos_compute += now.elapsed().as_nanos();
        }
        self.cache_cost
            .fetch_add(hlcvs.len() as u64, Ordering::SeqCst);
        self.compute_cost
            .fetch_add(hlcvs.len() as u64, Ordering::SeqCst);
        self.micros_cache
            .fetch_add((nanos_cache / 1000) as u64, Ordering::SeqCst);
        self.micros_compute
            .fetch_add((nanos_compute / 1000) as u64, Ordering::SeqCst);
        self.rank(
            timeframe,
            period as u32,
            hline as u32,
            lline as u32,
            &account,
        );
        account
    }

    pub fn random_search(&mut self) {
        let end = 1_0;
        let now = Instant::now();
        let i = AtomicI32::new(0);
        (0..end).into_iter().for_each(|_| {
            let timeframe = random::<u32>() % TIMEFRAME_RANGE * TIMEFRAME_STEP + MIN_TIMEFRAME;
            let len = random::<usize>() % MAX_LEN as usize;
            let hline = random::<u32>() % MAX_HLINE;
            let lline = random::<u32>() % MAX_LLINE;
            let hline = hline as f32;
            let lline = lline as f32;
            self.backtest(timeframe, len, hline, lline);
            let i = i.fetch_add(1, Ordering::SeqCst) + 1;
            println!(
                "{}/{}, {} combs/s",
                i,
                end,
                i as f32 / (now.elapsed().as_nanos() as f32 / 1e9)
            );
        });
    }

    //[sparring/src/main.rs:44] top10.iter().map(|x|
    //                x.balance).sorted_by_key(|x| OrderedFloat(*x)).collect_vec() = [
    //    128.72197,
    //    132.06981,
    //    133.41286,
    //    135.42238,
    //    143.17303,
    //    150.56244,
    //    152.15628,
    //    169.24411,
    //    171.70618,
    //    260.32178,
    //]
    //11634.352 - secs taken
    pub fn grid_search(&mut self) {
        let total = TIMEFRAME_RANGE as u64 * MAX_LEN as u64 * MAX_HLINE as u64 * MAX_LLINE as u64;
        for timeframe in (MIN_TIMEFRAME..MAX_TIMEFRAME).step_by(TIMEFRAME_STEP as usize) {
            let hlcvs = self.build_hlcvs(timeframe);
            for len in 2..MAX_LEN as usize {
                let mut cache = Vec::with_capacity(hlcvs.len());
                let mut state = RsiState::new(len, &hlcvs);
                let mut offset = state.update_at();
                for _ in 0..offset {
                    cache.push(0.0f32);
                }
                for i in offset..hlcvs.len() {
                    cache.push(state.update(&hlcvs, i));
                }
                self.cache_cost
                    .fetch_add(hlcvs.len() as u64, Ordering::SeqCst);
                offset += 1;
                let now = Instant::now();
                (0..MAX_HLINE).into_par_iter().for_each(|hline| {
                    let mut accounts = vec![Account::new(); MAX_LLINE as usize];
                    let mut prev_close = hlcvs[offset - 1].close;
                    let mut prev_rsi = cache[offset - 1];
                    let hline = hline as f32;
                    for i in offset..hlcvs.len() {
                        let rsi = cache[i];
                        for lline in 0..MAX_LLINE as usize {
                            let account = unsafe { accounts.get_unchecked_mut(lline) };
                            account.update(
                                &hlcvs,
                                i,
                                prev_close,
                                prev_rsi,
                                rsi,
                                self.max_risk,
                                hline,
                                lline as f32,
                            );
                        }
                        prev_close = hlcvs[i].close;
                        prev_rsi = rsi;
                    }
                    for i in 0..MAX_LLINE {
                        self.rank(
                            timeframe,
                            len as u32,
                            hline as u32,
                            i,
                            &accounts[i as usize],
                        );
                    }
                    self.compute_cost
                        .fetch_add(hlcvs.len() as u64 * MAX_LLINE as u64, Ordering::SeqCst);
                });
                println!(
                    "{} comb/s",
                    MAX_HLINE as f32 * MAX_LLINE as f32 / now.elapsed().as_nanos() as f32 * 1e9
                )
            }
        }
    }

    pub fn genetic_search(&self) {
        let mut param_count = 4;
        // Ako se koristi GA + BO algoritam
        if self.hybrid {
            param_count = 3;
        }
        let first_guess = vec![0.5; param_count];
        // Konfiguracija broja populacije i generacije
        let pop = 1000;
        let gen = 1000;
        println!("{pop}pop {gen}gen");

        // Konfiguriranje CMA-ES algoritma.
        let mut cmaes_state = CMAESOptions::new(first_guess, 0.5)
            .mode(Mode::Maximize)
            .population_size(pop)
            .max_generations(gen)
            // "self" (Context struktura) ima implementiran "trait" koji sadrži funkciju koja se
            // pokreće za svaku jedinku.
            .build(self)
            // Terminiranje programa ukoliko se pojavi neka greška.
            .unwrap();

        // Pokretanje optimizacije.
        cmaes_state.run();
    }

    pub fn bayes_search(&self) {
        // Objektivna funkcija.
        let f = |x: &[f64]| {
            let timeframe = MIN_TIMEFRAME + TIMEFRAME_STEP * x[0] as u32;
            let len = x[1] as usize;
            let hline = x[2] as u32 as f32;
            let lline = x[3] as u32 as f32;
            let account = self.backtest(timeframe, len, hline, lline);
            account.balance
        };
        // Definiranje područja pretraživanja.
        let input_interval = vec![
            (0., TIMEFRAME_RANGE as f64),
            (2., MAX_LEN as f64),
            (0., MAX_HLINE as f64),
            (0., MAX_LLINE as f64),
        ];
        // Broj iteracija optimizatora.
        let nb_iterations = 100;

        // Pokretanje optimizatora.
        Optimizer::new(&f, &input_interval, false)
            // "Optimizer" implementira "Iterator" "trait" te se preko njega pokreće.
            // Evaluiranje "nb_iterations" uzoraka.
            .skip(nb_iterations)
            // Traži se zadnja vrijednost kako bi se iterator izvršio.
            .next()
            .unwrap();
    }

    pub fn sharded_genetic_shearch(&mut self, n_shards: usize) {
        // Naznačuje se da se genetski algoritam mora podijeliti na 'n_shards' dijelova.
        self.n_shards = n_shards;
        let mut top = vec![Pair::default(); 10];
        for shard_i in 0..n_shards {
            self.shard_i = shard_i;
            // Izvršavanje genetske optimizacije nad manjem području vremenskog okvira.
            self.genetic_search();
            // Spremaju se najbolji parametri.
            let mut guard = self.top.lock();
            guard.sort_by_key(|x| OrderedFloat(x.balance));
            top.sort_by_key(|x| OrderedFloat(x.balance));
            let a: Vec<_> = guard.iter().map(|x| x.balance).collect();
            for ga_pair in 0..guard.len() {
                for pair in 0..top.len() {
                    if guard[ga_pair].balance > top[pair].balance {
                        top[pair] = guard[ga_pair].clone();
                        break;
                    }
                }
            }
            for result in guard.iter_mut() {
                *result = Pair::default();
            }
        }
        top.sort_by_key(|x| OrderedFloat(x.balance));
        self.top = Mutex::new(top);
        self.n_shards = 0;
    }

    pub fn hybrid_search(&mut self) {
        // Naznačuje se da se radi o hibridnom načinu rada genetskog algoritma.
        self.hybrid = true;
        let top = Mutex::new(vec![Pair::default(); 10]);
        // Funkcija koja se izvršava za svaku točku Bayesove optimizacije.
        let f = |x: &[f64]| {
            let timeframe = MIN_TIMEFRAME + TIMEFRAME_STEP * x[0] as u32;
            // Sprema se trenutni vremenski okvir.
            self.timeframe.store(timeframe, Ordering::SeqCst);
            // Izvšava se genetska optimizacija bez vremenskog okvira.
            self.genetic_search();
            // Spremaju se najbolje vrijednosti.
            let mut guard = self.top.lock();
            let ga_pair = guard
                .iter()
                .max_by_key(|x| OrderedFloat(x.balance))
                .unwrap();
            let mut top = top.lock();

            let (min_i, pair) = top
                .iter()
                .enumerate()
                .min_by_key(|x| OrderedFloat(x.1.balance))
                .unwrap();
            if ga_pair.balance > pair.balance {
                top[min_i] = ga_pair.clone();
            }
            let balance = ga_pair.balance;
            for result in guard.iter_mut() {
                *result = Pair::default();
            }
            balance
        };
        let input_interval = vec![(0., TIMEFRAME_RANGE as f64)];
        let nb_iterations = 30;

        // Kreiranje novog Simple(x) optimizatora i evauliranje 'nb_iterations' broj točaka.
        let (max_value, coordinates) = Optimizer::new(&f, &input_interval, false)
            .map(|x| {
                print!("hybrid {nb_iterations} ");
                x
            })
            .skip(nb_iterations)
            .next()
            .unwrap();
        self.top = top;
        self.hybrid = false;
    }

    pub fn summary(&self) {
        let compute = self.micros_compute.load(Ordering::SeqCst) as f32 / 1e6;
        let cache = self.micros_cache.load(Ordering::SeqCst) as f32 / 1e6;
        let timeframe = self.micros_timeframe.load(Ordering::SeqCst) as f32 / 1e6;
        println!(
            "compute_cost: {},
cache_cost: {},
timeframe_cost: {},
elapsed_compute_s: {},
elapsed_cache_s: {},
elapsed_timeframe_s: {},
elapsed_total_s: {},",
            self.compute_cost.load(Ordering::SeqCst),
            self.cache_cost.load(Ordering::SeqCst),
            self.timeframe_cost.load(Ordering::SeqCst),
            compute,
            cache,
            timeframe,
            compute + cache + timeframe
        );
        println!(
            "Top n: {:#?}",
            self.top
                .lock()
                .iter()
                .sorted_by_key(|x| OrderedFloat(x.balance))
        );
    }

    fn rank(&self, timeframe: u32, len: u32, hline: u32, lline: u32, account: &Account) {
        let mut top = self.top.lock();
        let (min_i, pair) = top
            .iter()
            .enumerate()
            .min_by_key(|x| OrderedFloat(x.1.balance))
            .unwrap();
        if account.balance > pair.balance {
            top[min_i] = Pair {
                params: [timeframe, len, hline, lline],
                balance: account.balance,
            }
        }
    }
}

// Implementacija osobine koja je potrebna za izvršavanje genetskog algoritma.
impl ObjectiveFunction for &Context {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        // Pozivanje parallelne implementacije.
        ParallelObjectiveFunction::evaluate_parallel(self, x)
    }
}

// Biblioteka koristi dvije osobine, jednu za izvršavanje na jednoj jezgri procesora i drugu na više.
// Struktura implementira obije zbog lakšeg razvoja programa. Da bi se algoritam izvodio paralelno
// potrebno je pozvati "run_parallel" metodu.
impl ParallelObjectiveFunction for &Context {
    fn evaluate_parallel(&self, orig: &DVector<f64>) -> f64 {
        let mut offset = 0;
        // U slučaju hibridnog načina rada se ignorira vremenski period.
        if self.hybrid {
            offset = 1;
        }
        for x in orig {
            if *x > 1. || *x < 0. {
                return 0.;
            }
        }
        // Orig je vektor brojeva između 0.0 i 1.0 on se skalira da bi stao u definirano područje
        // pretraživanja.
        let x = convert(orig, self.n_shards, self.shard_i);

        // Učitava se potrebni vremenski okvir u sljučaju GA + BO algoritma.
        let timeframe = if self.hybrid {
            self.timeframe.load(Ordering::SeqCst)
        } else {
            x[0].0 as u32
        };
        let period = x[1 - offset].0 as usize;
        let hline = x[2 - offset].0 as u32 as f32;
        let lline = x[3 - offset].0 as u32 as f32;
        let account = self.backtest(timeframe, period, hline, lline);
        account.balance as f64
    }
}

pub fn convert(x: &DVector<f64>, n_shards: usize, shard_i: usize) -> Vec<OrderedFloat<f64>> {
    let mut out = vec![OrderedFloat(0.); x.len()];
    let mut offset = 1;

    if x.len() == 4 {
        if n_shards == 1 {
            out[0] = OrderedFloat(
                MIN_TIMEFRAME as f64
                    + x[0].clamp(0., 1.) * TIMEFRAME_STEP as f64 * (TIMEFRAME_RANGE - 1) as f64,
            );
            out[0] = OrderedFloat((out[0].0 as u32 - out[0].0 as u32 % TIMEFRAME_STEP) as f64);
        } else {
            out[0] = OrderedFloat(
                MIN_TIMEFRAME as f64
                    + TIMEFRAME_STEP as f64 * TIMEFRAME_RANGE as f64 / n_shards as f64
                        * shard_i as f64
                    + x[0].clamp(0., 1.)
                        * TIMEFRAME_STEP as f64
                        * (TIMEFRAME_RANGE as usize / n_shards) as f64,
            );
            out[0] = OrderedFloat((out[0].0 as u32 - out[0].0 as u32 % TIMEFRAME_STEP) as f64);
        }
        offset = 0;
    }
    out[1 - offset] = OrderedFloat(2. + x[1 - offset].clamp(0., 1.) * (MAX_LEN - 2) as f64);
    out[2 - offset] = OrderedFloat(x[2 - offset].clamp(0., 1.) * MAX_HLINE as f64);
    out[3 - offset] = OrderedFloat(x[3 - offset].clamp(0., 1.) * MAX_LLINE as f64);

    out
}

#[derive(Default, Debug, Clone)]
pub struct Pair {
    pub params: [u32; 4],
    pub balance: f32,
}
