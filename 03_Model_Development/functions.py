"""Model data preparation, inference, losses and training utilities."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow_utils import add_sr_derived_rrs, add_spectral_indices, add_seasonal_features
from models import HierarchicalRrsChlaTransformer_ver2, MultiTaskLossWithUncertainty


# runtime
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def evaluation_seed(checkpoint, requested=None):
    """Reuse the training split; never silently evaluate a different L3 split."""
    metadata = checkpoint.get('metadata', {})
    recorded = metadata.get('seed')
    if recorded is None:
        recorded = metadata.get('hyperparams', {}).get('split_seed')
    if recorded is None:
        if requested is None:
            raise ValueError('Checkpoint has no split seed; provide the verified training seed with --seed')
        return int(requested)
    if requested is not None and int(requested) != int(recorded):
        raise ValueError(f'Evaluation seed {requested} differs from training seed {recorded}')
    return int(recorded)


def summarize_runs(folder):
    records = [{'run': path.parent.name, **json.loads(path.read_text())}
               for path in sorted(Path(folder).glob('run_*/summary.json'))]
    if not records:
        raise ValueError('No executed run summaries found')
    frame = pd.DataFrame(records).sort_values('validation_score', na_position='last')
    frame.to_csv(Path(folder)/'executed_search_results.csv', index=False)
    print(frame.to_string(index=False))
    eligible = frame.dropna(subset=['selected_epoch', 'validation_score'])
    if not eligible.empty:
        print('Validation-selected run:', eligible.iloc[0]['run'])
    return frame


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_config(path=None):
    return json.loads(Path(path or ROOT / '03_Model_Development/model_config.json').read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def make_model(hparams, scalers, columns, ocx_config=None):
    return HierarchicalRrsChlaTransformer_ver2(
        columns, d_model=hparams['md'], nhead=hparams['nh'],
        num_encoder_layers=hparams['nl'], dropout=hparams['dr'],
        dim_feedforward=hparams.get('fd', 4 * hparams['md']), scalers=scalers,
        ocx_config=ocx_config)


def load_model(checkpoint_path, device='cpu', expected_sha256=None):
    if expected_sha256 and sha256(checkpoint_path) != expected_sha256:
        raise ValueError('Checkpoint SHA-256 mismatch')
    # The official checkpoint includes sklearn scalers. Load only trusted files.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata']
    columns, scalers = metadata['columns_dict'], metadata['scalers']
    hparams = metadata.get('hyperparams', checkpoint.get('hyperparams'))
    if hparams is None:
        raise ValueError('Checkpoint is missing hyperparameters')
    model = make_model(hparams, scalers, columns, metadata.get('ocx_config'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    return model, scalers, columns, checkpoint


def prepare_features(frame):
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
    for operation in (add_sr_derived_rrs, add_spectral_indices, add_seasonal_features):
        frame = operation(frame)
    return frame.replace([np.inf, -np.inf], np.nan)


def predict_frame(frame, model, scalers, columns, device='cpu', batch_size=1024):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    features = prepare_features(frame)
    required = sum(columns['seq_input'], []) + columns['aux_input']
    positions = np.flatnonzero(features[required].notna().all(axis=1).to_numpy())
    raw = np.full(len(frame), np.nan, dtype=np.float32)
    log = np.full(len(frame), np.nan, dtype=np.float32)
    if len(positions) == 0:
        return raw, log
    valid = features.iloc[positions]
    seq = np.stack([valid[group].to_numpy(dtype=np.float32)
                    for group in columns['seq_input']], axis=-1)
    aux = scalers['aux_input'].transform(
        valid[columns['aux_input']].to_numpy(dtype=np.float32)).astype(np.float32)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(valid), batch_size):
            _, pred = model(torch.from_numpy(seq[start:start+batch_size]).to(device),
                            torch.from_numpy(aux[start:start+batch_size]).to(device))
            predictions.append(pred.cpu().numpy())
    values = scalers['target'].inverse_transform(np.concatenate(predictions)).reshape(-1)
    log[positions] = values
    raw[positions] = np.power(10.0, values)
    return raw, log


def atomic_checkpoint(data, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


# insitu_core
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

INSITU_OBSERVATION_KEYS = ["Hylak_id", "date", "MOD09_date", "lat", "lon", "source", "insitu_Chl-a"]

def _make_seq_array(frame, seq_groups):
    return np.stack(
        [frame[group].to_numpy(dtype=np.float32) for group in seq_groups],
        axis=-1,
    )

def _split_70_10_20(frame, seed):
    random_state = np.random.RandomState(seed)
    train_validation, test = train_test_split(
        frame,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.1 / 0.8,
        random_state=random_state,
        shuffle=True,
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )

def _make_l3_loaders(model_data, columns_dict, batch_size, seed):
    train_frame, validation_frame, test_frame = _split_70_10_20(
        model_data, seed
    )
    split_frames = {
        "train": train_frame,
        "validation": validation_frame,
        "test": test_frame,
    }
    scalers = {}
    tensors = {split: {} for split in split_frames}

    for key, columns in columns_dict.items():
        if key == "seq_input":
            for split, frame in split_frames.items():
                tensors[split][key] = torch.from_numpy(
                    _make_seq_array(frame, columns)
                )
            continue

        scaler = MinMaxScaler()
        scaler.fit(train_frame[columns].to_numpy())
        scalers[key] = scaler
        for split, frame in split_frames.items():
            transformed = scaler.transform(frame[columns].to_numpy())
            tensors[split][key] = torch.tensor(
                transformed, dtype=torch.float32
            )

    datasets = {
        split: TensorDataset(
            *[tensors[split][key] for key in columns_dict]
        )
        for split in split_frames
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
        ),
    }
    split_sizes = {
        split: len(frame) for split, frame in split_frames.items()
    }
    return loaders, scalers, split_sizes

def _make_insitu_observations(pixel_data):
    pixel_data = pixel_data.copy()
    pixel_data["date"] = pd.to_datetime(pixel_data["date"])
    pixel_data["MOD09_date"] = pd.to_datetime(pixel_data["MOD09_date"])

    observations = (
        pixel_data[INSITU_OBSERVATION_KEYS]
        .drop_duplicates()
        .sort_values(INSITU_OBSERVATION_KEYS[:-1])
        .reset_index(drop=True)
    )
    observations["observation_id"] = np.arange(
        len(observations), dtype=np.int64
    )
    observations["target_log"] = np.log10(
        observations["insitu_Chl-a"].to_numpy(dtype=float)
    )
    if not np.isfinite(observations["target_log"]).all():
        raise ValueError("In-situ Chl-a must be positive and finite.")

    pixel_data = pixel_data.merge(
        observations[INSITU_OBSERVATION_KEYS + ["observation_id"]],
        on=INSITU_OBSERVATION_KEYS,
        how="inner",
        validate="many_to_one",
    )
    return pixel_data, observations

def _validate_manifest_observations(manifest, observations):
    expected_ids = set(observations["observation_id"])
    actual_ids = set(manifest["observation_id"])
    if expected_ids != actual_ids:
        raise ValueError(
            "Existing in-situ split manifest does not match the current "
            "observation dataset."
        )
    if manifest["observation_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicated observations.")

    expected = observations[
        ["observation_id"] + INSITU_OBSERVATION_KEYS
    ].copy()
    actual = manifest[
        ["observation_id"] + INSITU_OBSERVATION_KEYS
    ].copy()
    for column in ["date", "MOD09_date"]:
        expected[column] = pd.to_datetime(expected[column])
        actual[column] = pd.to_datetime(actual[column])
    expected = expected.sort_values("observation_id").reset_index(drop=True)
    actual = actual.sort_values("observation_id").reset_index(drop=True)
    if not expected.equals(actual):
        raise ValueError(
            "Existing in-situ split manifest observation metadata does not "
            "match the current dataset."
        )

def _make_bag_tensors(
    split_pixels,
    split_observations,
    columns_dict,
    scalers,
    max_pixels,
):
    observation_ids = split_observations["observation_id"].to_numpy(
        dtype=np.int64
    )
    observation_position = {
        observation_id: position
        for position, observation_id in enumerate(observation_ids)
    }
    seq_groups = columns_dict["seq_input"]
    seq_length = len(seq_groups[0])
    seq_channels = len(seq_groups)
    aux_columns = columns_dict["aux_input"]
    number_of_observations = len(split_observations)

    seq_bags = np.zeros(
        (
            number_of_observations,
            max_pixels,
            seq_length,
            seq_channels,
        ),
        dtype=np.float32,
    )
    aux_bags = np.zeros(
        (number_of_observations, max_pixels, len(aux_columns)),
        dtype=np.float32,
    )
    masks = np.zeros(
        (number_of_observations, max_pixels),
        dtype=bool,
    )

    for observation_id, pixels in split_pixels.groupby(
        "observation_id", sort=False
    ):
        position = observation_position[int(observation_id)]
        pixel_count = len(pixels)
        if pixel_count > max_pixels:
            raise ValueError(
                f"observation_id={observation_id} has {pixel_count} pixels; "
                f"max_pixels={max_pixels}."
            )
        seq_bags[position, :pixel_count] = _make_seq_array(
            pixels, seq_groups
        )
        aux_bags[position, :pixel_count] = scalers["aux_input"].transform(
            pixels[aux_columns].to_numpy()
        ).astype(np.float32)
        masks[position, :pixel_count] = True

    if (~masks.any(axis=1)).any():
        raise ValueError("At least one in-situ observation has no pixels.")

    targets_log = (
        split_observations.set_index("observation_id")
        .loc[observation_ids, "target_log"]
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )
    return TensorDataset(
        torch.from_numpy(seq_bags),
        torch.from_numpy(aux_bags),
        torch.from_numpy(masks),
        torch.from_numpy(targets_log),
        torch.from_numpy(observation_ids),
    )

def _insitu_bag_loss(
    model,
    batch,
    criterion,
    device,
    observation_weight_by_id=None,
):
    seq_bag, aux_bag, pixel_mask, target_log, observation_id = batch
    seq_bag = seq_bag.to(device)
    aux_bag = aux_bag.to(device)
    pixel_mask = pixel_mask.to(device)
    target_log = target_log.to(device)

    batch_size, max_pixels = pixel_mask.shape
    flat_mask = pixel_mask.reshape(-1)
    flat_seq = seq_bag.reshape(
        batch_size * max_pixels,
        seq_bag.shape[2],
        seq_bag.shape[3],
    )[flat_mask]
    flat_aux = aux_bag.reshape(
        batch_size * max_pixels,
        aux_bag.shape[2],
    )[flat_mask]
    observation_indices = (
        torch.arange(batch_size, device=device)
        .unsqueeze(1)
        .expand(batch_size, max_pixels)
        .reshape(-1)[flat_mask]
    )

    _, pixel_prediction_scaled = model(flat_seq, flat_aux)
    target_min = model.target_min.to(device).reshape(1, -1)
    target_max = model.target_max.to(device).reshape(1, -1)
    target_range = target_max - target_min + 1e-8
    pixel_prediction_log = (
        pixel_prediction_scaled * target_range + target_min
    )
    pixel_prediction_raw = torch.pow(
        10.0,
        torch.clamp(pixel_prediction_log, min=-6.0, max=6.0),
    )

    raw_sums = torch.zeros(
        (batch_size, 1),
        dtype=pixel_prediction_raw.dtype,
        device=device,
    )
    raw_sums.index_add_(
        0,
        observation_indices,
        pixel_prediction_raw,
    )
    counts = pixel_mask.sum(dim=1, keepdim=True).to(
        pixel_prediction_raw.dtype
    )
    bag_prediction_raw = raw_sums / counts.clamp_min(1.0)
    bag_prediction_log = torch.log10(
        bag_prediction_raw.clamp_min(1e-6)
    )

    bag_prediction_scaled = (
        bag_prediction_log - target_min
    ) / target_range
    target_scaled = (target_log - target_min) / target_range
    if observation_weight_by_id is None:
        loss = criterion(bag_prediction_scaled, target_scaled)
    else:
        per_observation_loss = (
            bag_prediction_scaled - target_scaled
        ).pow(2).mean(dim=1)
        observation_weights = torch.as_tensor(
            [
                observation_weight_by_id[int(value)]
                for value in observation_id.tolist()
            ],
            dtype=per_observation_loss.dtype,
            device=device,
        )
        loss = (per_observation_loss * observation_weights).mean()
    return loss, bag_prediction_log, target_log

def _validate_l3(model, dataloader, criterion, loss_fn, device, phase):
    totals = {"loss": 0.0, "mid_rrs": 0.0, "target": 0.0}
    with torch.no_grad():
        for seq_input, aux_input, y_mid_rrs, y_target in dataloader:
            seq_input = seq_input.to(device)
            aux_input = aux_input.to(device)
            y_mid_rrs = y_mid_rrs.to(device)
            y_target = y_target.to(device)
            pred_mid_rrs, pred_target = model(seq_input, aux_input)
            loss_mid_rrs = criterion(pred_mid_rrs, y_mid_rrs)
            loss_target = criterion(pred_target, y_target)
            if phase == "rrs_warmup":
                loss = loss_mid_rrs
            else:
                loss, _ = loss_fn(
                    [loss_mid_rrs, loss_target],
                    task_indices=[0, 1],
                )
            totals["loss"] += loss.item()
            totals["mid_rrs"] += loss_mid_rrs.item()
            totals["target"] += loss_target.item()
    batches = len(dataloader)
    return {key: value / batches for key, value in totals.items()}

def _validate_insitu_with_lake_macro(
    model,
    dataloader,
    device,
    lake_id_by_observation,
    recalibration_lake_ids=None,
):
    """Return exact observation MSE and equal-lake-weighted macro MSE."""
    observation_ids = []
    squared_errors = []
    criterion = torch.nn.MSELoss()
    target_range = (
        model.target_max.to(device) - model.target_min.to(device) + 1e-8
    ).reshape(1, -1)

    with torch.no_grad():
        for batch in dataloader:
            _, predictions_log, targets_log = _insitu_bag_loss(
                model, batch, criterion, device
            )
            batch_squared_errors = (
                (predictions_log - targets_log) / target_range
            ).pow(2).mean(dim=1)
            observation_ids.append(batch[-1].cpu())
            squared_errors.append(batch_squared_errors.cpu())

    observation_ids = torch.cat(observation_ids).numpy().astype(np.int64)
    squared_errors = torch.cat(squared_errors).numpy().astype(float)
    validation = pd.DataFrame({
        "observation_id": observation_ids,
        "squared_error": squared_errors,
    })
    validation["Hylak_id"] = validation["observation_id"].map(
        lake_id_by_observation
    )
    if validation["Hylak_id"].isna().any():
        missing = validation.loc[
            validation["Hylak_id"].isna(), "observation_id"
        ].tolist()
        raise KeyError(
            "Missing Hylak_id for validation observations: "
            f"{missing[:10]}"
        )
    lake_mse = validation.groupby("Hylak_id")["squared_error"].mean()
    recalibration_lake_ids = set(recalibration_lake_ids or [])
    recalibration_mask = validation["Hylak_id"].isin(
        recalibration_lake_ids
    )

    def _group_mean(mask):
        values = validation.loc[mask, "squared_error"]
        return float(values.mean()) if len(values) else float("nan")

    return {
        "observation_mse": float(validation["squared_error"].mean()),
        "recal_observation_mse": _group_mean(recalibration_mask),
        "nonrecal_observation_mse": _group_mean(~recalibration_mask),
        "lake_macro_mse": float(lake_mse.mean()),
        "lake_count": int(len(lake_mse)),
        "observation_count": int(len(validation)),
        "recal_observation_count": int(recalibration_mask.sum()),
        "nonrecal_observation_count": int((~recalibration_mask).sum()),
    }



# data
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader



def insitu_loaders(pixel_data, manifest_path, columns, scalers, batch_size=128):
    pixels = pixel_data.loc[(pixel_data['insitu_Chl-a'] > 0)
                            & (pixel_data['insitu_Chl-a'] < 1000)].copy()
    pixels = pixels.drop(columns=['observation_id', 'split'], errors='ignore')
    pixels, observations = _make_insitu_observations(pixels)
    manifest = pd.read_parquet(manifest_path)
    for name in ('date', 'MOD09_date'):
        manifest[name] = pd.to_datetime(manifest[name])
    _validate_manifest_observations(manifest, observations)
    if set(manifest['split']) != {'train', 'validation', 'test'}:
        raise ValueError('Manifest must contain train, validation and test')
    if manifest.groupby('Hylak_id')['split'].nunique().max() != 1:
        raise ValueError('A lake appears in more than one split')
    sizes = pixels.groupby('observation_id').size()
    if not sizes.between(5, 9).all():
        raise ValueError('Each observation must have 5–9 valid pixels')
    required = sum(columns['seq_input'], []) + columns['aux_input']
    if not np.isfinite(pixels[required].to_numpy(dtype=float)).all():
        raise ValueError('In-situ inputs contain nonfinite values')
    # Compute targets from observations, never trust a stale transformed column.
    manifest['target_log'] = np.log10(manifest['insitu_Chl-a'].to_numpy(dtype=float))
    loaders = {}
    for split in ('train', 'validation', 'test'):
        obs = manifest[manifest['split'] == split].sort_values('observation_id')
        pix = pixels[pixels['observation_id'].isin(obs['observation_id'])]
        ds = _make_bag_tensors(pix, obs, columns, scalers, max_pixels=9)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'))
    return loaders, manifest


def training_data(l3_path, pixels_path, manifest_path, columns, batch_size, insitu_batch_size, seed):
    l3 = pd.read_parquet(l3_path)
    # Input is the curated/recalibrated L3-TD, not unprocessed NASA Chl-a.
    if (l3['Chl-a'] <= 0).any():
        raise ValueError('L3 Chl-a must be positive')
    l3['log_Chl-a'] = np.log10(l3['Chl-a'])
    required = sum(columns['seq_input'], []) + columns['aux_input'] + columns['mid_rrs'] + columns['target']
    if not np.isfinite(l3[required].to_numpy(dtype=float)).all():
        raise ValueError('L3-TD contains nonfinite inputs or targets; curate before splitting')
    l3_loaders, scalers, _ = _make_l3_loaders(l3, columns, batch_size, seed)
    bags, manifest = insitu_loaders(pd.read_parquet(pixels_path), manifest_path,
                                   columns, scalers, insitu_batch_size)
    return l3_loaders, bags, scalers, manifest


# metrics
import numpy as np


def chla_metrics(observed, predicted):
    observed, predicted = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted) & (observed > 0) & (predicted > 0)
    x, y = observed[valid], predicted[valid]
    if not len(x):
        return dict(n=0, r_log=np.nan, rmse_log=np.nan, r2_log=np.nan, pbias_raw=np.nan)
    lx, ly = np.log10(x), np.log10(y)
    denominator = np.square(lx-lx.mean()).sum()
    return dict(n=len(x),
        r_log=float(np.corrcoef(lx, ly)[0, 1]) if len(x)>1 and lx.std()>0 and ly.std()>0 else np.nan,
        rmse_log=float(np.sqrt(np.mean(np.square(ly-lx)))),
        r2_log=float(1-np.square(ly-lx).sum()/denominator) if denominator>0 else np.nan,
        pbias_raw=float(100*(x-y).sum()/x.sum()))


# training
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import importlib.metadata
from sklearn.model_selection import ParameterGrid



def grid_candidates(configuration):
    """Cartesian grid; an execution record is created only by train_run."""
    candidates = []
    for hp in ParameterGrid(configuration['grid']):
        hp = dict(hp)
        if hp['md'] % hp['nh']:
            continue
        hp['fd'] = hp['md'] * hp.pop('ff_factor')
        candidates.append(hp)
    return candidates


def train_run(config, hp, args, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite a run: {output}')
    set_seed(args.seed)
    columns = config['columns_dict']
    settings = {**config['hyperparams'], **hp}
    settings['split_seed'] = args.seed
    device = args.device
    l3, bags, scalers, manifest = training_data(
        args.l3, args.insitu, args.manifest, columns, hp['bs'],
        settings['insitu_batch_size'], args.seed)
    ocx_config = config.get('ocx_config')
    model = make_model(hp, scalers, columns, ocx_config=ocx_config).to(device)
    weights = MultiTaskLossWithUncertainty(3).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(weights.parameters()), lr=hp['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=15)
    output.mkdir(parents=True)
    metadata = {'hyperparams': settings, 'columns_dict': columns, 'scalers': scalers,
                'ocx_config': ocx_config,
                'seed': args.seed, 'training_validation_metric': config['training_validation_metric'],
                'final_epoch_validation_metric': config['final_epoch_validation_metric'],
                'input_sha256': {name: sha256(getattr(args, name)) for name in ('l3', 'insitu', 'manifest')},
                'package_versions': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'pandas', 'scikit-learn')},
                'implementation': 'portable-public-training-loop'}
    (output / 'run_config.json').write_text(json.dumps(
        {key: value for key, value in metadata.items() if key != 'scalers'}, indent=2))
    lake_by_obs = manifest.set_index('observation_id')['Hylak_id'].to_dict()
    history = []
    target_best = monitor_best = candidate_best = float('inf')
    target_wait = monitor_wait = 0
    stage = 'target'
    chosen_epoch = None
    alpha = settings['selection_insitu_alpha']
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        model.train(); weights.train()
        phase = ('rrs_warmup' if epoch <= settings['rrs_warmup_epochs'] else
                 'l3_warmup' if epoch <= settings['l3_warmup_epochs'] else 'joint')
        iterator = iter(bags['train'])
        totals = np.zeros(3, dtype=float)
        n_insitu = 0
        for step, batch in enumerate(l3['train'], 1):
            seq, aux, rrs_true, target_true = [item.to(device) for item in batch]
            optimizer.zero_grad()
            rrs, target = model(seq, aux)
            rrs_loss, target_loss = criterion(rrs, rrs_true), criterion(target, target_true)
            if phase == 'rrs_warmup':
                objective = rrs_loss
            else:
                losses, task_ids = [rrs_loss, target_loss], [0, 1]
                if phase == 'joint' and step % settings['insitu_update_interval'] == 0:
                    try:
                        bag = next(iterator)
                    except StopIteration:
                        iterator = iter(bags['train']); bag = next(iterator)
                    loss, _, _ = _insitu_bag_loss(model, bag, criterion, device)
                    losses.append(loss); task_ids.append(2)
                    totals[2] += loss.item(); n_insitu += 1
                objective, _ = weights(losses, task_indices=task_ids)
            if not torch.isfinite(objective):
                raise FloatingPointError(f'Nonfinite loss: epoch={epoch}, batch={step}')
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite gradient: epoch={epoch}, batch={step}; no optimizer step')
            optimizer.step()
            totals[:2] += [rrs_loss.item(), target_loss.item()]
        model.eval(); weights.eval()
        val = _validate_l3(model, l3['validation'], criterion, weights, device, phase)
        insitu = _validate_insitu_with_lake_macro(model, bags['validation'], device, lake_by_obs)
        target_value = val['target']
        candidate = target_value + alpha * insitu['observation_mse']
        monitoring = target_value + alpha * insitu[config['training_validation_metric']]
        if not np.isfinite([target_value, candidate, monitoring]).all():
            raise FloatingPointError(f'Nonfinite validation metric at epoch {epoch}')
        eligible = False
        if epoch >= settings['selection_start_epoch']:
            if stage == 'target':
                if target_value < target_best - settings['target_min_delta']:
                    target_best, target_wait = target_value, 0
                else:
                    target_wait += 1
                if target_wait >= settings['target_stage_patience'] or epoch >= settings['force_selection_epoch']:
                    stage = 'joint'
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=15)
            if stage == 'joint':
                eligible = True
                if monitoring < monitor_best - settings['selection_min_delta']:
                    monitor_best, monitor_wait = monitoring, 0
                else:
                    monitor_wait += 1
            scheduler.step(monitoring if stage == 'joint' else target_value)
        row = {'epoch': epoch, 'phase': phase, 'selection_stage': stage,
               'train_loss_mid_rrs': totals[0]/len(l3['train']),
               'train_loss_target': totals[1]/len(l3['train']),
               'train_loss_insitu': totals[2]/n_insitu if n_insitu else np.nan,
               'val_loss_target': target_value, 'val_loss_mid_rrs': val['mid_rrs'],
               'val_loss_insitu_observation': insitu['observation_mse'],
               'val_loss_insitu_lake_macro': insitu['lake_macro_mse'],
               'val_selection_loss': candidate, 'selection_eligible': eligible,
               'insitu_updates': n_insitu, 'epoch_seconds': time.monotonic()-start,
               'lr': optimizer.param_groups[0]['lr']}
        history.append(row)
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                      'loss_fn_state_dict': weights.state_dict(), 'metadata': metadata,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}
        atomic_checkpoint(checkpoint, output / f'epoch_{epoch:03d}.pth')
        if eligible and candidate < candidate_best:
            candidate_best, chosen_epoch = candidate, epoch
            atomic_checkpoint(checkpoint, output / 'selected_model.pth')
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(f'epoch={epoch} L3={target_value:.6f} in-situ={insitu["observation_mse"]:.6f} stage={stage}', flush=True)
        if stage == 'joint' and monitor_wait >= settings['selection_stage_patience']:
            break
    result = {**hp, 'selected_epoch': chosen_epoch,
              'validation_score': candidate_best if chosen_epoch else None,
              'status': 'completed' if chosen_epoch else 'no_selection_stage',
              'epochs_completed': epoch}
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    return result
